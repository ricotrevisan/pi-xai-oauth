import { describe, expect, it, vi } from "vitest";

const { primeStreamSimple } = vi.hoisted(() => ({
  primeStreamSimple: vi.fn((model: any) => {
    const message = {
      role: "assistant",
      content: [{ type: "text", text: "Prime delegate" }],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: "stop",
      timestamp: Date.now(),
    };
    return {
      async *[Symbol.asyncIterator]() {
        yield { type: "done", reason: "stop", message };
      },
      result: () => Promise.resolve(message),
    };
  }),
}));

vi.mock("@earendil-works/pi-ai", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai")>()),
  streamSimple: primeStreamSimple,
}));

vi.mock("@earendil-works/pi-ai/compat", () => {
  throw new Error("Prime Agent does not export the inherited /compat subpath");
});

describe("Prime Agent Responses compatibility", () => {
  it("streams through the host root delegate without resolving /compat", async () => {
    const { streamSimpleXaiResponses } = await import(
      "../../extensions/xai/responses"
    );
    const stream = streamSimpleXaiResponses(
      {
        id: "grok-4.6",
        name: "Grok 4.6",
        provider: "xai-auth",
        api: "xai-responses",
        baseUrl: "https://cli-chat-proxy.grok.com/v1",
        headers: {},
        reasoning: true,
        input: ["text", "image"],
        cost: { input: 2, output: 6, cacheRead: 0.5, cacheWrite: 0 },
        contextWindow: 500_000,
        maxTokens: 131_072,
      } as any,
      { messages: [] } as any,
      { apiKey: "oauth-token", sessionId: "prime-session" } as any,
    );

    await expect(stream.result()).resolves.toMatchObject({
      api: "xai-responses",
      provider: "xai-auth",
      model: "grok-4.6",
      stopReason: "stop",
    });
    expect(primeStreamSimple).toHaveBeenCalledOnce();
    expect(primeStreamSimple.mock.calls[0][0]).toMatchObject({
      api: "openai-responses",
      provider: "xai-auth",
      id: "grok-4.6",
    });
  });
});
