import { Link } from 'react-router';

export default function CompletePage() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 py-12 text-white">
      <div className="w-full max-w-sm space-y-6 text-center">
        <div className="space-y-3">
          <div className="text-5xl">✓</div>
          <h1 className="text-2xl font-bold">Recording Saved</h1>
          <p className="text-sm text-gray-400">
            Your recording has been uploaded and analyzed.
          </p>
        </div>
        <Link
          to="/"
          className="flex min-h-12 w-full items-center justify-center rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
        >
          Record Another
        </Link>
      </div>
    </div>
  );
}
