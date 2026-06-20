// Native Anthropic API backend (Claude models).
//
// Uses the official @anthropic-ai/sdk Messages API — NOT an OpenAI-compatible
// shim. Claude models (model names starting with "anthropic/" or "claude") are
// routed here first when ANTHROPIC_API_KEY is set; call_llm.ts falls back to the
// OpenAI/OpenRouter endpoint if the native call is unavailable or errors, so
// "anthropic/claude-..." strings keep working for OpenRouter users with no key.
//
// Model id: any "anthropic/" prefix is stripped to the native id (e.g.
// "anthropic/claude-haiku-4-5" -> "claude-haiku-4-5").
import Anthropic from "@anthropic-ai/sdk";
import { buildSystemPrompt, PromptOptions } from "./prompt.ts";
import type { ApiRetryOptions, CodeReturn, ConfirmResult, Usage } from "./call_llm.ts";

const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_TIMEOUT_MS = 600000;
// The Messages API requires max_tokens (unlike the OpenAI path). Generous enough
// for a repl block + brief reasoning; overridable via llm_kwargs.max_tokens.
const DEFAULT_MAX_TOKENS = 16000;

export function anthropicApiKey(): string | undefined {
    return Deno.env.get("RLM_ANTHROPIC_API_KEY") || Deno.env.get("ANTHROPIC_API_KEY") || undefined;
}

// Optional override of the Anthropic endpoint (proxy/gateway). When unset, the
// SDK uses the default https://api.anthropic.com.
export function anthropicBaseURL(): string | undefined {
    return Deno.env.get("RLM_ANTHROPIC_BASE_URL") || Deno.env.get("ANTHROPIC_BASE_URL") || undefined;
}

export function isAnthropicModel(model: string): boolean {
    return model.startsWith("anthropic/") || model.startsWith("claude");
}

export function stripAnthropicPrefix(model: string): string {
    return model.replace(/^anthropic\//, "");
}

// fast-rlm messages are OpenAI-shaped {role, content}. Anthropic takes the system
// prompt as a top-level param (built here, like the other backends) and a
// user/assistant message list with string content.
// deno-lint-ignore no-explicit-any
function toAnthropicMessages(messages: any[]): any[] {
    return messages
        .filter((m) => m && m.role !== "system")
        .map((m) => ({
            role: m.role === "assistant" ? "assistant" : "user",
            content: typeof m.content === "string" ? m.content : JSON.stringify(m.content ?? ""),
        }));
}

function extractReplCode(content: string): string {
    const replMatches = [...content.matchAll(/```repl([\s\S]*?)```/g)];
    return replMatches.map((m) => m[1].trim()).join("\n");
}

// Map Anthropic's split usage onto fast-rlm's Usage. Anthropic separates uncached
// (input_tokens) from cached (cache_read/creation); prompt_tokens is the sum so it
// matches the OpenAI path's semantics. Cost is not returned by the SDK.
// deno-lint-ignore no-explicit-any
function mapUsage(u: any): Usage {
    const cacheRead = u?.cache_read_input_tokens ?? 0;
    const cacheCreate = u?.cache_creation_input_tokens ?? 0;
    const input = u?.input_tokens ?? 0;
    const output = u?.output_tokens ?? 0;
    return {
        prompt_tokens: input + cacheRead + cacheCreate,
        completion_tokens: output,
        total_tokens: input + cacheRead + cacheCreate + output,
        cached_tokens: cacheRead,
        reasoning_tokens: 0,
        cost: undefined,
    };
}

// One Messages API turn: build system + messages, call Claude, return text + usage.
async function anthropicComplete(
    messages: any[], // deno-lint-ignore-line no-explicit-any
    model_name: string,
    is_leaf_agent: boolean,
    options: ApiRetryOptions | undefined,
    promptOpts: PromptOptions | undefined,
    llmKwargs: Record<string, unknown> | null | undefined,
): Promise<{ text: string; usage: Usage }> {
    const baseURL = anthropicBaseURL();
    const client = new Anthropic({
        apiKey: anthropicApiKey(),
        maxRetries: options?.maxRetries ?? DEFAULT_MAX_RETRIES,
        timeout: options?.timeout ?? DEFAULT_TIMEOUT_MS,
        ...(baseURL ? { baseURL } : {}),
    });

    // max_tokens is required by the API and not part of the OpenAI llm_kwargs
    // convention, so pull it out of llmKwargs if present and default otherwise.
    const kwargs: Record<string, unknown> = { ...(llmKwargs ?? {}) };
    const max_tokens = typeof kwargs.max_tokens === "number" ? kwargs.max_tokens as number : DEFAULT_MAX_TOKENS;
    delete kwargs.max_tokens;

    const resp = await client.messages.create({
        model: stripAnthropicPrefix(model_name),
        max_tokens,
        system: buildSystemPrompt(is_leaf_agent, promptOpts ?? {}),
        messages: toAnthropicMessages(messages),
        ...kwargs,
        // deno-lint-ignore no-explicit-any
    } as any);

    // deno-lint-ignore no-explicit-any
    const text = (resp.content as any[])
        .filter((b) => b.type === "text")
        .map((b) => b.text)
        .join("");
    return { text, usage: mapUsage(resp.usage) };
}

// Drop-in for generate_code when model_name is a Claude model with a key set.
// Throws on failure so call_llm.ts can fall back to the OpenAI/OpenRouter path.
export async function generateAnthropicCode(
    messages: any[], // deno-lint-ignore-line no-explicit-any
    model_name: string,
    is_leaf_agent = false,
    options?: ApiRetryOptions,
    promptOpts?: PromptOptions,
    llmKwargs?: Record<string, unknown> | null,
): Promise<CodeReturn> {
    const { text, usage } = await anthropicComplete(messages, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
    const code = extractReplCode(text);
    const message = { role: "assistant", content: text };
    return { code, success: !!code, message, usage };
}

// Drop-in for confirmDelegation (compression guard) when model_name is Claude.
export async function confirmAnthropicDelegation(
    baseMessages: any[], // deno-lint-ignore-line no-explicit-any
    confirmQuestion: string,
    model_name: string,
    is_leaf_agent: boolean,
    options?: ApiRetryOptions,
    promptOpts?: PromptOptions,
    llmKwargs?: Record<string, unknown> | null,
): Promise<ConfirmResult> {
    const messages = [...baseMessages, { role: "user", content: confirmQuestion }];
    const { text, usage } = await anthropicComplete(messages, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
    const content = text.trim();
    // Fail-open: only an explicit "NO" (as the first word) rejects.
    const firstWord = content.replace(/^[^a-zA-Z]+/, "").slice(0, 4).toUpperCase();
    const approve = !firstWord.startsWith("NO");
    return { approve, reason: content || "(no reason given)", usage };
}
