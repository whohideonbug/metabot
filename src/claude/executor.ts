import { execSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import { query } from '@anthropic-ai/claude-agent-sdk';
import type { SDKUserMessage, SpawnOptions, SpawnedProcess } from '@anthropic-ai/claude-agent-sdk';
import type { BotConfigBase } from '../config.js';
import type { Logger } from '../utils/logger.js';
import { AsyncQueue } from '../utils/async-queue.js';

const isWindows = process.platform === 'win32';

/** Resolve the Claude Code binary path at module load time. */
function resolveClaudePath(): string {
  if (process.env.CLAUDE_EXECUTABLE_PATH) return process.env.CLAUDE_EXECUTABLE_PATH;
  try {
    const cmd = isWindows ? 'where claude' : 'which claude';
    return execSync(cmd, { encoding: 'utf-8' }).trim().split(/\r?\n/)[0];
  } catch {
    return isWindows ? 'claude' : '/usr/local/bin/claude';
  }
}

const CLAUDE_EXECUTABLE = resolveClaudePath();

/**
 * Env var prefixes to always strip from the inherited process environment.
 * CLAUDE*: prevents "nested session" errors from the SDK.
 */
const ALWAYS_FILTERED_PREFIXES = ['CLAUDE'];

/**
 * Auth-related env vars that are only filtered when an explicit API key
 * is provided in bots.json OR when ~/.claude/.credentials.json exists.
 * This ensures users who rely solely on ANTHROPIC_API_KEY env var can
 * still authenticate without configuring bots.json.
 */
const AUTH_ENV_VARS = ['ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN'];

/**
 * Check if Claude Code has credentials.json (OAuth login).
 */
function hasCredentialsFile(): boolean {
  const credPath = path.join(os.homedir(), '.claude', '.credentials.json');
  try {
    return fs.existsSync(credPath);
  } catch {
    return false;
  }
}

/**
 * Create a custom spawn function for cross-platform compatibility.
 * - Uses process.execPath (current Node binary) to avoid PATH issues on Windows.
 * - Always filters CLAUDE* env vars to prevent nested session errors.
 * - Filters ANTHROPIC auth env vars only when an explicit API key is provided
 *   or credentials.json exists (so env-var-only users can still authenticate).
 * - Merges process.env so child inherits system PATH, TEMP, etc.
 * - Optionally injects an explicit ANTHROPIC_API_KEY from bots.json config.
 */
function createSpawnFn(explicitApiKey?: string): (options: SpawnOptions) => SpawnedProcess {
  // Decide once whether to filter auth env vars
  const filterAuthVars = !!(explicitApiKey || hasCredentialsFile());

  return (options: SpawnOptions): SpawnedProcess => {
    const nodePath = process.execPath;

    // Merge provided env with process.env for a complete environment
    const baseEnv = options.env && Object.keys(options.env).length > 0
      ? { ...process.env, ...options.env }
      : { ...process.env };

    // Filter out env vars that interfere with auth or cause nested session errors
    const env: Record<string, string> = {};
    for (const [key, value] of Object.entries(baseEnv)) {
      if (value === undefined) continue;
      if (ALWAYS_FILTERED_PREFIXES.some(p => key.startsWith(p))) continue;
      if (filterAuthVars && AUTH_ENV_VARS.some(v => key.startsWith(v))) continue;
      env[key] = value;
    }

    // Inject explicit API key from bots.json (after filtering, so it takes effect)
    if (explicitApiKey) {
      env.ANTHROPIC_API_KEY = explicitApiKey;
    }

    const child = spawn(nodePath, options.args, {
      cwd: options.cwd,
      env,
      signal: options.signal,
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    return child as unknown as SpawnedProcess;
  };
}

export interface ApiContext {
  botName: string;
  chatId: string;
  /** Group chat member names — enables inter-bot communication prompt. */
  groupMembers?: string[];
  /** Group ID — used to build grouptalk chatIds for inter-bot communication. */
  groupId?: string;
}

export interface ExecutorOptions {
  prompt: string;
  cwd: string;
  sessionId?: string;
  abortController: AbortController;
  outputsDir?: string;
  apiContext?: ApiContext;
  /** Override maxTurns for this execution. */
  maxTurns?: number;
  /** Override model for this execution (e.g. faster model for voice calls). */
  model?: string;
  /** Override allowed tools for this execution (empty array = no tools). */
  allowedTools?: string[];
}

export type SDKMessage = {
  type: string;
  subtype?: string;
  uuid?: string;
  session_id?: string;
  message?: {
    content?: Array<{
      type: string;
      text?: string;
      name?: string;
      id?: string;
      input?: unknown;
    }>;
  };
  // Result fields
  duration_ms?: number;
  duration_api_ms?: number;
  total_cost_usd?: number;
  result?: string;
  is_error?: boolean;
  num_turns?: number;
  errors?: string[];
  // Model usage from result message (per-model breakdown)
  modelUsage?: Record<string, { inputTokens: number; outputTokens: number; contextWindow: number; costUSD: number }>;
  // Stream event fields
  event?: {
    type: string;
    index?: number;
    delta?: {
      type: string;
      text?: string;
    };
    content_block?: {
      type: string;
      text?: string;
      name?: string;
      id?: string;
    };
  };
  parent_tool_use_id?: string | null;
};

export interface ExecutionHandle {
  stream: AsyncGenerator<SDKMessage>;
  sendAnswer(toolUseId: string, sessionId: string, answerText: string): void;
  finish(): void;
}

export class ClaudeExecutor {
  constructor(
    private config: BotConfigBase,
    private logger: Logger,
  ) {}

  /** Shared system-prompt sections for SDK `query()` and `cliExecute()`. */
  private buildAppendSections(outputsDir?: string, apiContext?: ApiContext): string[] {
    const appendSections: string[] = [];

    if (outputsDir) {
      appendSections.push(`## Output Files\nWhen producing output files for the user (images, PDFs, documents, archives, code files, etc.), copy them to: ${outputsDir}\nUse \`cp\` via the Bash tool. The bridge will automatically send files placed there to the user.`);
    }

    if (apiContext) {
      // botName and chatId are per-session — inject into system prompt to avoid
      // race conditions when multiple chats run concurrently.
      // Port and secret are already set as METABOT_* env vars in config.ts.
      appendSections.push(
        `## MetaBot API\nYou are running as bot "${apiContext.botName}" in chat "${apiContext.chatId}".\nUse the /metabot skill for full API documentation (agent bus, scheduling, bot management).`
      );

      // Group chat — tell the bot who else is in the group and how to talk to them
      if (apiContext.groupMembers && apiContext.groupMembers.length > 0) {
        const others = apiContext.groupMembers.filter((m) => m !== apiContext.botName);
        const groupId = apiContext.groupId;
        if (groupId) {
          appendSections.push(
            `## Group Chat\nYou are in a group chat (group: ${groupId}) with these bots: ${others.join(', ')}.\nTo talk to another bot, use: \`mb talk <botName> grouptalk-${groupId}-<botName> "message"\`\nExample: \`mb talk ${others[0]} grouptalk-${groupId}-${others[0]} "hello"\`\nIMPORTANT: Always use the grouptalk-${groupId}-<botName> chatId pattern when talking to other bots in this group.`
          );
        } else {
          appendSections.push(
            `## Group Chat\nYou are in a group chat with these bots: ${others.join(', ')}.\nUse \`mb talk <botName> <chatId> "message"\` to communicate with other bots in the group.`
          );
        }
      }
    }

    return appendSections;
  }

  /** Environment for a direct `claude` subprocess (inherits `process.env`, optional API key from bot config). */
  private buildCliEnv(): Record<string, string> {
    const env: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined) env[k] = v;
    }
    if (this.config.claude.apiKey) {
      env.ANTHROPIC_API_KEY = this.config.claude.apiKey;
    }
    return env;
  }

  /**
   * Args for `claude -p` (print mode) with stream-json, aligned with `execute()` / `buildQueryOptions`
   * (same append sections as SDK path; no `apiContext` — same as `execute()`).
   */
  private buildCliArgs(prompt: string, sessionId: string | undefined, outputsDir?: string): string[] {
    const args: string[] = [
      '-p',
      '--output-format', 'stream-json',
      '--include-partial-messages',
      '--permission-mode', 'bypassPermissions',
      '--dangerously-skip-permissions',
      '--setting-sources', 'user,project',
      '--betas', 'context-1m-2025-08-07',
    ];
    const appendSections = this.buildAppendSections(outputsDir, undefined);
    if (appendSections.length > 0) {
      args.push('--append-system-prompt', appendSections.join('\n\n'));
    }
    if (sessionId) {
      args.push('--resume', sessionId);
    }
    if (this.config.claude.model) {
      args.push('--model', this.config.claude.model);
    }
    if (this.config.claude.maxBudgetUsd !== undefined) {
      args.push('--max-budget-usd', String(this.config.claude.maxBudgetUsd));
    }
    args.push(prompt);
    return args;
  }

  private buildQueryOptions(cwd: string, sessionId: string | undefined, abortController: AbortController, outputsDir?: string, apiContext?: ApiContext): Record<string, unknown> {
    const queryOptions: Record<string, unknown> = {
      permissionMode: 'bypassPermissions' as const,
      allowDangerouslySkipPermissions: true,
      cwd,
      abortController,
      includePartialMessages: true,
      // Load MCP servers and settings from user/project config files
      settingSources: ['user', 'project'],
      // Cross-platform spawn: custom spawn filters CLAUDE* env vars and uses
      // process.execPath to avoid PATH issues on Windows; fileURLToPath converts
      // file:// URLs to native paths for the SDK CLI entrypoint.
      spawnClaudeCodeProcess: createSpawnFn(this.config.claude.apiKey),
      executableArgs: [path.join(path.dirname(fileURLToPath(import.meta.resolve('@anthropic-ai/claude-agent-sdk'))), 'cli.js')],
      pathToClaudeCodeExecutable: CLAUDE_EXECUTABLE,
    };

    const appendSections = this.buildAppendSections(outputsDir, apiContext);
    if (appendSections.length > 0) {
      queryOptions.systemPrompt = {
        type: 'preset',
        preset: 'claude_code',
        append: '\n\n' + appendSections.join('\n\n'),
      };
    }

    if (this.config.claude.maxTurns !== undefined) {
      queryOptions.maxTurns = this.config.claude.maxTurns;
    }

    if (this.config.claude.maxBudgetUsd !== undefined) {
      queryOptions.maxBudgetUsd = this.config.claude.maxBudgetUsd;
    }

    if (this.config.claude.model) {
      queryOptions.model = this.config.claude.model;
    }

    if (sessionId) {
      queryOptions.resume = sessionId;
    }

    // Beta flags are ignored by the SDK on OAuth/Pro-Max auth. For 1M context,
    // use the model-name suffix `[1m]` (e.g. `claude-opus-4-7[1m]`) instead.
    queryOptions.betas = ['context-1m-2025-08-07'];

    return queryOptions;
  }

  /**
   * Default request entrypoint: tries the environment `claude` CLI first (single-turn, `-p` mode),
   * and only falls back to the SDK `query()` multi-turn path if the CLI fails before producing
   * any progress. When CLI succeeds, `sendAnswer`/`finish` become no-ops (CLI is single-turn);
   * when the stream falls back to the SDK path, they are wired to the SDK input queue as before.
   */
  startExecution(options: ExecutorOptions): ExecutionHandle {
    const { cwd, sessionId, abortController, outputsDir } = options;

    this.logger.info(
      { mode: 'cli-first', cwd, hasSession: !!sessionId, outputsDir },
      '[claude] startExecution: trying CLI first, SDK query() multi-turn fallback',
    );

    const logger = this.logger;
    const cliStream = this.cliExecute(options);

    let sdkHandle: ExecutionHandle | null = null;
    const self = this;

    async function* combinedStream(): AsyncGenerator<SDKMessage> {
      let yieldedProgress = false;
      let fallbackReason: string | null = null;

      try {
        for await (const msg of cliStream) {
          if (!yieldedProgress && msg.type === 'result' && msg.is_error) {
            fallbackReason = msg.result || msg.errors?.join('; ') || 'cli result error';
            break;
          }
          yield msg;
          if (
            msg.type === 'system' ||
            msg.type === 'assistant' ||
            msg.type === 'stream_event' ||
            (msg.type === 'result' && !msg.is_error)
          ) {
            yieldedProgress = true;
          }
        }
      } catch (err: any) {
        if (err?.name === 'AbortError' || abortController.signal.aborted) {
          return;
        }
        if (yieldedProgress) {
          throw err;
        }
        fallbackReason = String(err?.message ?? err);
      }

      if (fallbackReason !== null && !yieldedProgress && !abortController.signal.aborted) {
        logger.warn({ mode: 'cli->sdk', reason: fallbackReason }, '[claude] CLI failed with no progress, falling back to SDK query() multi-turn');
        sdkHandle = self.startSdkExecution(options);
        yield* sdkHandle.stream;
      }
    }

    return {
      stream: combinedStream(),
      sendAnswer: (toolUseId: string, sid: string, answerText: string) => {
        if (sdkHandle) {
          sdkHandle.sendAnswer(toolUseId, sid, answerText);
        } else {
          logger.warn({ mode: 'cli', toolUseId }, '[claude] sendAnswer ignored (CLI mode is single-turn)');
        }
      },
      finish: () => {
        if (sdkHandle) sdkHandle.finish();
      },
    };
  }

  /**
   * Original SDK `query()`-based multi-turn execution (preserved). Exposed so the bridge can call
   * it directly when needed, and used internally by `startExecution()` as the CLI fallback.
   */
  startSdkExecution(options: ExecutorOptions): ExecutionHandle {
    const { prompt, cwd, sessionId, abortController, outputsDir, apiContext } = options;

    this.logger.info(
      {
        mode: 'sdk',
        cwd,
        hasSession: !!sessionId,
        outputsDir,
        promptChars: prompt.length,
        model: options.model ?? this.config.claude.model,
        maxTurns: options.maxTurns ?? this.config.claude.maxTurns,
        allowedTools: options.allowedTools,
      },
      '[claude] invoking SDK query() (multi-turn)',
    );

    const inputQueue = new AsyncQueue<SDKUserMessage>();

    // Push the initial user message
    const initialMessage: SDKUserMessage = {
      type: 'user',
      message: {
        role: 'user' as const,
        content: prompt,
      },
      parent_tool_use_id: null,
      session_id: sessionId || '',
    };
    inputQueue.enqueue(initialMessage);

    const queryOptions = this.buildQueryOptions(cwd, sessionId, abortController, outputsDir, apiContext);
    if (options.maxTurns !== undefined) {
      queryOptions.maxTurns = options.maxTurns;
    }
    if (options.model) {
      queryOptions.model = options.model;
    }
    if (options.allowedTools !== undefined) {
      queryOptions.allowedTools = options.allowedTools;
    }

    const stream = query({
      prompt: inputQueue,
      options: queryOptions as any,
    });

    const logger = this.logger;

    async function* wrapStream(): AsyncGenerator<SDKMessage> {
      // Race each stream.next() against the abort signal so we exit immediately on /stop
      const abortPromise = new Promise<never>((_, reject) => {
        if (abortController.signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'));
          return;
        }
        abortController.signal.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        }, { once: true });
      });

      const iterator = stream[Symbol.asyncIterator]();

      try {
        while (true) {
          const result = await Promise.race([
            iterator.next(),
            abortPromise,
          ]);
          if (result.done) break;
          yield result.value as SDKMessage;
        }
      } catch (err: any) {
        if (err.name === 'AbortError' || abortController.signal.aborted) {
          logger.info('Claude execution aborted');
          // Clean up the underlying iterator (non-blocking)
          try { iterator.return?.(undefined); } catch { /* ignore */ }
          return;
        }
        throw err;
      }
    }

    return {
      stream: wrapStream(),
      sendAnswer: (toolUseId: string, sid: string, answerText: string) => {
        logger.info({ toolUseId }, 'Sending answer to Claude');
        const answerMessage: SDKUserMessage = {
          type: 'user',
          message: {
            role: 'user' as const,
            content: [
              {
                type: 'tool_result',
                tool_use_id: toolUseId,
                content: answerText,
              },
            ],
          },
          parent_tool_use_id: null,
          session_id: sid,
        };
        inputQueue.enqueue(answerMessage);
      },
      finish: () => {
        inputQueue.finish();
      },
    };
  }

  async *execute(options: ExecutorOptions): AsyncGenerator<SDKMessage> {
    const { prompt, cwd, sessionId, abortController, outputsDir } = options;

    this.logger.info(
      {
        mode: 'sdk',
        cwd,
        hasSession: !!sessionId,
        promptChars: prompt.length,
        model: this.config.claude.model,
      },
      '[claude] invoking SDK query() (single-turn)',
    );

    const queryOptions = this.buildQueryOptions(cwd, sessionId, abortController, outputsDir);

    const stream = query({
      prompt,
      options: queryOptions as any,
    });

    const abortPromise = new Promise<never>((_, reject) => {
      if (abortController.signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      abortController.signal.addEventListener('abort', () => {
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    });

    const iterator = stream[Symbol.asyncIterator]();

    try {
      while (true) {
        const result = await Promise.race([
          iterator.next(),
          abortPromise,
        ]);
        if (result.done) break;
        yield result.value as SDKMessage;
      }
    } catch (err: any) {
      if (err.name === 'AbortError' || abortController.signal.aborted) {
        this.logger.info('Claude execution aborted');
        try { iterator.return?.(undefined); } catch { /* ignore */ }
        return;
      }
      throw err;
    }
  }

  /**
   * Same contract as `execute()` but runs the environment `claude` binary with `-p` and `--output-format stream-json`,
   * parsing NDJSON lines into `SDKMessage` (for `StreamProcessor`). No Agent SDK subprocess.
   */
  async *cliExecute(options: ExecutorOptions): AsyncGenerator<SDKMessage> {
    const { prompt, cwd, sessionId, abortController, outputsDir } = options;

    if (abortController.signal.aborted) {
      return;
    }

    const args = this.buildCliArgs(prompt, sessionId, outputsDir);
    // Strip the trailing prompt when logging args to avoid dumping user input
    const argsForLog = args.slice(0, -1);

    this.logger.info(
      {
        mode: 'cli',
        claudePath: CLAUDE_EXECUTABLE,
        cwd,
        hasSession: !!sessionId,
        promptChars: prompt.length,
        model: this.config.claude.model,
        args: argsForLog,
      },
      '[claude] invoking CLI (claude -p stream-json)',
    );

    // Pre-flight: if we resolved an absolute path that no longer exists
    // (common with nvm node-version switches), short-circuit to an error
    // result so `startExecution` can fall back to the SDK path instead of
    // crashing on an uncaught ENOENT from spawn.
    if (path.isAbsolute(CLAUDE_EXECUTABLE) && !fs.existsSync(CLAUDE_EXECUTABLE)) {
      const msg = `claude binary not found at ${CLAUDE_EXECUTABLE} (set CLAUDE_EXECUTABLE_PATH to override)`;
      this.logger.warn({ mode: 'cli', claudePath: CLAUDE_EXECUTABLE }, `[claude] ${msg}`);
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: msg,
        errors: [msg],
      } as SDKMessage;
      return;
    }

    let child: ReturnType<typeof spawn>;
    try {
      child = spawn(CLAUDE_EXECUTABLE, args, {
        cwd,
        env: this.buildCliEnv(),
        signal: abortController.signal,
        stdio: ['ignore', 'pipe', 'pipe'],
      });
    } catch (err: any) {
      const msg = String(err?.message ?? err);
      this.logger.warn({ mode: 'cli', claudePath: CLAUDE_EXECUTABLE, err }, '[claude] spawn threw synchronously');
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: msg,
        errors: [msg],
      } as SDKMessage;
      return;
    }

    // Attach 'error' listener SYNCHRONOUSLY (before any await) so async spawn
    // failures (ENOENT, EACCES, etc.) never escape as unhandled events and
    // crash the process. Captured here and surfaced as an error-result so the
    // fallback path in `startExecution` can take over.
    let spawnError: Error | null = null;
    child.on('error', (err: Error) => {
      spawnError = err;
      this.logger.warn({ mode: 'cli', claudePath: CLAUDE_EXECUTABLE, err }, '[claude] CLI spawn error');
    });

    this.logger.debug({ mode: 'cli', pid: child.pid }, '[claude] CLI process spawned');

    let stderr = '';
    child.stderr?.on('data', (chunk: unknown) => {
      stderr += String(chunk);
    });

    const abortPromise = new Promise<never>((_, reject) => {
      if (abortController.signal.aborted) {
        reject(new DOMException('Aborted', 'AbortError'));
        return;
      }
      abortController.signal.addEventListener('abort', () => {
        reject(new DOMException('Aborted', 'AbortError'));
      }, { once: true });
    });

    // Give any synchronous-style spawn error (e.g. ENOENT) a tick to surface
    // via the 'error' listener before we start awaiting stdout.
    await new Promise<void>((resolve) => setImmediate(resolve));
    if (spawnError) {
      const msg = String((spawnError as Error).message ?? spawnError);
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: msg,
        errors: [msg],
      } as SDKMessage;
      return;
    }

    const rl = readline.createInterface({
      input: child.stdout!,
      crlfDelay: Infinity,
    });

    try {
      const iterator = rl[Symbol.asyncIterator]();
      while (true) {
        const result = await Promise.race([iterator.next(), abortPromise]);
        if (result.done) break;
        const line = result.value as string;
        const trimmed = line.trim();
        if (!trimmed) continue;
        try {
          yield JSON.parse(trimmed) as SDKMessage;
        } catch {
          this.logger.debug({ line: trimmed.slice(0, 120) }, '[claude] CLI: skip non-JSON line');
        }
      }
    } catch (err: any) {
      if (err.name === 'AbortError' || abortController.signal.aborted) {
        this.logger.info({ mode: 'cli', pid: child.pid }, '[claude] CLI execution aborted');
        return;
      }
      throw err;
    } finally {
      rl.close();
    }

    // 'error' may fire mid-stream (e.g. child crashes). Surface as terminal
    // error-result so fallback path can trigger.
    if (spawnError) {
      const msg = String((spawnError as Error).message ?? spawnError);
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: msg,
        errors: [msg],
      } as SDKMessage;
      return;
    }

    let exitCode: number | null = null;
    try {
      const [code] = await once(child, 'exit');
      exitCode = code;
    } catch (err: any) {
      if (err.name === 'AbortError' || abortController.signal.aborted) {
        return;
      }
      this.logger.warn({ mode: 'cli', pid: child.pid, err }, '[claude] CLI process error');
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: String(err?.message ?? err),
        errors: [String(err?.message ?? err)],
      } as SDKMessage;
      return;
    }

    if (exitCode !== 0 && exitCode !== null && !abortController.signal.aborted) {
      this.logger.warn({ mode: 'cli', pid: child.pid, exitCode, stderr: stderr.slice(-2000) }, '[claude] CLI non-zero exit');
      yield {
        type: 'result',
        subtype: 'error',
        is_error: true,
        result: stderr || `claude exited with code ${exitCode}`,
        errors: [stderr || `exit ${exitCode}`],
      } as SDKMessage;
    } else {
      this.logger.info({ mode: 'cli', pid: child.pid, exitCode }, '[claude] CLI execution finished');
    }
  }
}
