/**
 * Parses the admin_token cookie from a request.
 */
function getAdminTokenCookie(request: Request): string | null {
  const cookieHeader = request.headers.get("Cookie") ?? "";
  for (const part of cookieHeader.split(";")) {
    const [name, ...rest] = part.trim().split("=");
    if (name === "admin_token") return rest.join("=");
  }
  return null;
}

type AuthResult =
  | { ok: true }
  | { redirect: Response }
  | { unauthorized: true };

/**
 * Checks admin authentication from cookie or ?token= query param.
 *
 * - If ?token=X matches adminToken → set cookie + return redirect to clean URL
 * - If cookie matches adminToken → return { ok: true }
 * - Otherwise → return { unauthorized: true }
 */
export function checkAdminAuth(
  request: Request,
  adminToken: string,
): AuthResult {
  const url = new URL(request.url);
  const queryToken = url.searchParams.get("token");

  if (queryToken !== null) {
    if (queryToken === adminToken) {
      url.searchParams.delete("token");
      return {
        redirect: new Response(null, {
          status: 302,
          headers: {
            Location: url.pathname + (url.search || ""),
            "Set-Cookie": `admin_token=${adminToken}; HttpOnly; Path=/; SameSite=Strict`,
          },
        }),
      };
    }
    return { unauthorized: true };
  }

  const cookieToken = getAdminTokenCookie(request);
  if (cookieToken === adminToken) {
    return { ok: true };
  }

  return { unauthorized: true };
}
