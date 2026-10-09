// Thin proxy: POST a Clef request body (model, state, questions, images) and get the
// Clef answer back. Lets the Python app use Clef via the Workers AI binding, with no
// Cloudflare API token on the client side.
//
// Auth: requests must send "Authorization: Bearer <PROXY_TOKEN>" (set with
// `wrangler secret put PROXY_TOKEN`). For local `wrangler dev`, put ALLOW_NO_AUTH=true
// in .dev.vars instead. With neither configured, every request is refused.

const MODELS = new Set(["clef-flash"]);
const MAX_BODY_BYTES = 13 * 1024 * 1024; // Clef's documented request limit

const json = (data, status = 200) =>
  new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });

function authorised(request, env) {
  if (env.PROXY_TOKEN) return request.headers.get("authorization") === `Bearer ${env.PROXY_TOKEN}`;
  return env.ALLOW_NO_AUTH === "true";
}

export default {
  async fetch(request, env) {
    if (request.method !== "POST") return json({ error: "POST a Clef request body" }, 405);
    if (!authorised(request, env)) return json({ error: "unauthorised" }, 401);
    if (Number(request.headers.get("content-length") || 0) > MAX_BODY_BYTES) {
      return json({ error: "request too large" }, 413);
    }

    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "body must be JSON" }, 400);
    }
    const model = typeof body.model === "string" ? body.model.trim() : "";
    if (!MODELS.has(model)) return json({ error: `model must be one of ${[...MODELS].join(", ")}` }, 400);

    try {
      const result = await env.AI.run(`@cf/cloudflare/${model}`, body);
      return json(result);
    } catch (err) {
      return json({ error: String(err && err.message ? err.message : err) }, 502);
    }
  },
};
