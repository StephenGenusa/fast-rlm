// Unit test: the `instruction` PromptOption is appended verbatim to the system
// prompt of both root and leaf agents, and omitted when absent/empty.
//
// Run:  deno test --allow-read --allow-env tests/instruction_test.ts
import { assert, assertStringIncludes } from "jsr:@std/assert@^1.0.0";
import { buildSystemPrompt } from "../src/prompt.ts";

const WRAPPER = "Here is the user's instructions - you must follow it closely:";

Deno.test("instruction appended to root prompt", () => {
    const p = buildSystemPrompt(false, { instruction: "FOO_DIRECTIVE" });
    assertStringIncludes(p, WRAPPER);
    assertStringIncludes(p, "FOO_DIRECTIVE");
    // appended at the very end
    assert(p.trimEnd().endsWith("FOO_DIRECTIVE"));
});

Deno.test("instruction appended to leaf prompt", () => {
    const p = buildSystemPrompt(true, { instruction: "FOO_DIRECTIVE" });
    assertStringIncludes(p, WRAPPER);
    assertStringIncludes(p, "FOO_DIRECTIVE");
});

Deno.test("no instruction -> no wrapper", () => {
    assert(!buildSystemPrompt(false, {}).includes(WRAPPER));
    assert(!buildSystemPrompt(true, {}).includes(WRAPPER));
});

Deno.test("empty / whitespace instruction -> no wrapper", () => {
    assert(!buildSystemPrompt(false, { instruction: "" }).includes(WRAPPER));
    assert(!buildSystemPrompt(false, { instruction: "   \n  " }).includes(WRAPPER));
    assert(!buildSystemPrompt(false, { instruction: null }).includes(WRAPPER));
});
