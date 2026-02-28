import { PostHogProvider } from "@posthog/react";
import posthog from "posthog-js";
import { startTransition, StrictMode } from "react";
import { hydrateRoot } from "react-dom/client";
import { HydratedRouter } from "react-router/dom";
import LogRocket from "logrocket";

posthog.init(import.meta.env.VITE_PUBLIC_POSTHOG_TOKEN, {
  api_host: import.meta.env.VITE_PUBLIC_POSTHOG_HOST,
  defaults: "2025-05-24",
  capture_exceptions: true,
});

if (import.meta.env.PROD && import.meta.env.VITE_PUBLIC_LOGROCKET_APP_ID) {
  LogRocket.init(import.meta.env.VITE_PUBLIC_LOGROCKET_APP_ID);
}

startTransition(() => {
  hydrateRoot(
    document,
    <PostHogProvider client={posthog}>
      <StrictMode>
        <HydratedRouter />
      </StrictMode>
    </PostHogProvider>,
  );
});
