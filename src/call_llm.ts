import { OpenAI } from "openai";
import chalk from "npm:chalk@5";
import { buildSystemPrompt, PromptOptions } from "./prompt.ts";
import { createVertexClient, refreshVertexClient, isVertexModel, stripVertexPrefix } from "./vertex.ts";
import { confirmAcpDelegation, generateAcpCode, isAcpModel } from "./acp.ts";
import { anthropicApiKey, confirmAnthropicDelegation, generateAnthropicCode, isAnthropicModel } from "./anthropic.ts";

const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_TIMEOUT_MS = 600000;

export interface ApiRetryOptions {
    maxRetries?: number;
    timeout?: number;
}

export interface Usage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    cached_tokens: number;
    reasoning_tokens: number;
    cost: number | undefined;
}

export interface CodeReturn {
    code: string;
    success: boolean;
    message: any;
    usage: Usage;
}

// The default backend is any OpenAI-compatible API (OpenAI, DeepSeek, OpenRouter,
// or anything else via RLM_MODEL_BASE_URL) — OpenRouter is just the default base
// URL, not a requirement. The key resolves from RLM_MODEL_API_KEY, then
// OPENAI_API_KEY, then OPENROUTER_API_KEY.
const apiKey = Deno.env.get("RLM_MODEL_API_KEY") || Deno.env.get("OPENAI_API_KEY") || Deno.env.get("OPENROUTER_API_KEY");
const baseURL = Deno.env.get("RLM_MODEL_BASE_URL") || "https://openrouter.ai/api/v1";

const vertexMode = Deno.env.get("RLM_VERTEX_AI") === "1";

// Credentials are validated per backend at the point of use, not eagerly at
// startup — a run that only uses ACP agents needs no API key at all. The
// OpenAI-compatible path calls this when it's the backend actually selected;
// Vertex (ADC) and native Anthropic (ANTHROPIC_API_KEY) validate their own.
function requireModelApiKey(): string {
    if (!apiKey) {
        throw new Error(
            "No API key for the OpenAI-compatible backend. Set RLM_MODEL_API_KEY " +
            "(or OPENAI_API_KEY) to your key, e.g.: export RLM_MODEL_API_KEY='sk-...'. " +
            "Point it at any OpenAI-compatible endpoint (OpenAI, DeepSeek, OpenRouter, …) " +
            "via RLM_MODEL_BASE_URL.\n" +
            "For Vertex AI, set RLM_VERTEX_AI=1 and GOOGLE_CLOUD_PROJECT instead.\n" +
            "For Anthropic models, set ANTHROPIC_API_KEY instead.\n" +
            "(ACP agents such as acp:opencode need no API key.)",
        );
    }
    return apiKey;
}

let vertexClient: OpenAI | null = null;

export async function generate_code(
    messages: any[],
    model_name: string,
    is_leaf_agent: boolean = false,
    options?: ApiRetryOptions,
    promptOpts?: PromptOptions,
    // Arbitrary extra params (temperature, top_p, seed, ...) spread into the
    // chat.completions.create call. Passed end-to-end from run(llm_kwargs=...).
    llmKwargs?: Record<string, unknown> | null
): Promise<CodeReturn> {
    // ACP agents (e.g. "acp:codex") are a separate backend — see acp.ts.
    if (isAcpModel(model_name)) {
        return generateAcpCode(messages, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
    }

    // Claude models prefer the native Anthropic API (anthropic.ts) when a key is
    // set; on any failure fall through to the OpenAI/OpenRouter path below with the
    // original model string. Explicit Vertex (prefix or RLM_VERTEX_AI) wins.
    if (isAnthropicModel(model_name) && anthropicApiKey() && !isVertexModel(model_name) && !vertexMode) {
        try {
            return await generateAnthropicCode(messages, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
        } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            console.error(chalk.yellow(`⚠ Anthropic endpoint unavailable (${msg}); falling back to ${baseURL}`));
        }
    }

    const maxRetries = options?.maxRetries ?? DEFAULT_MAX_RETRIES;
    const timeout = options?.timeout ?? DEFAULT_TIMEOUT_MS;

    let client: OpenAI;
    let resolvedModel = model_name;

    if (isVertexModel(model_name) || vertexMode) {
        resolvedModel = isVertexModel(model_name) ? stripVertexPrefix(model_name) : model_name;
        if (!vertexClient) {
            vertexClient = await createVertexClient({ maxRetries, timeout });
        } else {
            vertexClient = await refreshVertexClient(vertexClient, { maxRetries, timeout });
        }
        client = vertexClient;
    } else {
        client = new OpenAI({
            apiKey: requireModelApiKey(),
            baseURL,
            maxRetries: maxRetries,
            timeout: timeout,
        });
    }

    try {
        // deno-lint-ignore no-explicit-any
        const createParams: any = {
            model: resolvedModel,
            messages: [
                { role: "system", content: buildSystemPrompt(is_leaf_agent, promptOpts ?? {}) },
                ...messages
            ],
            ...(llmKwargs ?? {}),
        };
        const completion = await client.chat.completions.create(createParams);

        const content = completion.choices[0].message.content || "";

        const replMatches = [...content.matchAll(/```repl([\s\S]*?)```/g)];
        let code = replMatches.map(m => m[1].trim()).join("\n");

        const usage: Usage = {
            prompt_tokens: completion.usage?.prompt_tokens ?? 0,
            completion_tokens: completion.usage?.completion_tokens ?? 0,
            total_tokens: completion.usage?.total_tokens ?? 0,
            cached_tokens: completion.usage?.prompt_tokens_details?.cached_tokens ?? 0,
            reasoning_tokens: completion.usage?.completion_tokens_details?.reasoning_tokens ?? 0,
            cost: (completion.usage as any)?.cost ?? undefined,
        };

        if (!code) {
            return {
                code: "",
                success: false,
                message: completion.choices[0].message,
                usage,
            };
        }

        return {
            code,
            success: true,
            message: completion.choices[0].message,
            usage,
        };
    } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        console.error(chalk.red(`✖ API call failed: ${msg}`));
        throw error;
    }
}

export interface ConfirmResult {
    approve: boolean;
    reason: string;
    usage: Usage;
}

/**
 * Compression-guard self-check. Reuses the subagent's exact opening prefix
 * (same system prompt + same probe messages) so the provider KV-cache is shared
 * with the real run, then appends a YES/NO confirmation question. Parsing is
 * fail-open: anything not clearly starting with "NO" is treated as approval.
 */
export async function confirmDelegation(
    baseMessages: any[],
    confirmQuestion: string,
    model_name: string,
    is_leaf_agent: boolean,
    options?: ApiRetryOptions,
    promptOpts?: PromptOptions,
    llmKwargs?: Record<string, unknown> | null
): Promise<ConfirmResult> {
    if (isAcpModel(model_name)) {
        return confirmAcpDelegation(baseMessages, confirmQuestion, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
    }

    if (isAnthropicModel(model_name) && anthropicApiKey() && !isVertexModel(model_name) && !vertexMode) {
        try {
            return await confirmAnthropicDelegation(baseMessages, confirmQuestion, model_name, is_leaf_agent, options, promptOpts, llmKwargs);
        } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            console.error(chalk.yellow(`⚠ Anthropic endpoint unavailable (${msg}); falling back to ${baseURL}`));
        }
    }

    let client: OpenAI;
    let resolvedModel = model_name;

    if (isVertexModel(model_name) || vertexMode) {
        resolvedModel = isVertexModel(model_name) ? stripVertexPrefix(model_name) : model_name;
        if (!vertexClient) {
            vertexClient = await createVertexClient({
                maxRetries: options?.maxRetries ?? DEFAULT_MAX_RETRIES,
                timeout: options?.timeout ?? DEFAULT_TIMEOUT_MS,
            });
        } else {
            vertexClient = await refreshVertexClient(vertexClient, {
                maxRetries: options?.maxRetries ?? DEFAULT_MAX_RETRIES,
                timeout: options?.timeout ?? DEFAULT_TIMEOUT_MS,
            });
        }
        client = vertexClient;
    } else {
        client = new OpenAI({
            apiKey: requireModelApiKey(),
            baseURL,
            maxRetries: options?.maxRetries ?? DEFAULT_MAX_RETRIES,
            timeout: options?.timeout ?? DEFAULT_TIMEOUT_MS,
        });
    }

    // deno-lint-ignore no-explicit-any
    const createParams: any = {
        model: resolvedModel,
        messages: [
            { role: "system", content: buildSystemPrompt(is_leaf_agent, promptOpts ?? {}) },
            ...baseMessages,
            { role: "user", content: confirmQuestion },
        ],
        ...(llmKwargs ?? {}),
    };
    const completion = await client.chat.completions.create(createParams);
    const content = (completion.choices[0].message.content || "").trim();

    const usage: Usage = {
        prompt_tokens: completion.usage?.prompt_tokens ?? 0,
        completion_tokens: completion.usage?.completion_tokens ?? 0,
        total_tokens: completion.usage?.total_tokens ?? 0,
        cached_tokens: completion.usage?.prompt_tokens_details?.cached_tokens ?? 0,
        reasoning_tokens: completion.usage?.completion_tokens_details?.reasoning_tokens ?? 0,
        cost: (completion.usage as any)?.cost ?? undefined,
    };

    // Fail-open: only an explicit "NO" (as the first word) rejects.
    const firstWord = content.replace(/^[^a-zA-Z]+/, "").slice(0, 4).toUpperCase();
    const approve = !firstWord.startsWith("NO");
    return { approve, reason: content || "(no reason given)", usage };
}

if (import.meta.main) {
    // Test with a dummy context
    const query_context = "Just return fibonacci sequence";
    const out = await generate_code([
        { "role": "user", "content": query_context },
    ], "gpt-5-mini"
    );
    console.log(out)

}

