import { Outlet, Link } from "react-router";
import type { Route } from "./+types/admin";
import { checkAdminAuth } from "~/lib/admin-auth";

export async function loader({ request, context }: Route.LoaderArgs) {
  const adminToken = context.cloudflare.env.ADMIN_TOKEN;
  const result = checkAdminAuth(request, adminToken);

  if ("redirect" in result) return result.redirect;
  if ("unauthorized" in result) throw new Response("Unauthorized", { status: 401 });

  return null;
}

export default function AdminLayout() {
  return (
    <div className="min-h-screen w-full bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-4 py-3">
        <div className="mx-auto flex max-w-5xl items-center gap-3">
          <Link
            to="/admin"
            className="text-sm font-semibold text-gray-200 hover:text-white transition"
          >
            Admin
          </Link>
        </div>
      </header>
      <main className="mx-auto max-w-5xl px-4 py-6">
        <Outlet />
      </main>
    </div>
  );
}
