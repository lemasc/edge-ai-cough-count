import { Link } from "react-router";

export default function App() {
  return (
    <div className="w-full max-w-sm space-y-8 flex flex-col justify-center">
      <div className="text-center">
        <div className="mb-4 text-5xl">🎤</div>
        <h1 className="text-2xl font-bold">Cough Dataset Collector</h1>
        <p className="mt-2 text-gray-400 text-sm">
          Records audio for cough detection research
        </p>
      </div>
      <Link
        to="/record"
        className="block text-center min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
      >
        Begin
      </Link>
    </div>
  );
}
