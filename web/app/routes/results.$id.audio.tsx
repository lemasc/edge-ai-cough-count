import type { Route } from './+types/results.$id.audio';
import { getDb } from '~/db';
import * as schema from '~/db/schema';
import { eq } from 'drizzle-orm';

export async function loader({ params, context }: Route.LoaderArgs) {
  const { env } = context.cloudflare;
  const db = getDb(env.DB);

  const rec = await db.query.recordings.findFirst({
    where: eq(schema.recordings.id, params.id),
  });

  if (!rec?.audioKey) throw new Response('Not found', { status: 404 });

  const obj = await env.STORAGE.get(rec.audioKey);
  if (!obj) throw new Response('Not found', { status: 404 });

  const headers = new Headers();
  obj.writeHttpMetadata(headers);
  headers.set('cache-control', 'public, max-age=31536000, immutable');

  return new Response(obj.body, { headers });
}
