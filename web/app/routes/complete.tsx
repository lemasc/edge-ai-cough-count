import { Link } from "react-router";
import type { Route } from "./+types/complete";

export function loader({ context }: Route.LoaderArgs) {
  return { surveyFormUrl: context.cloudflare.env.SURVEY_FORM_URL };
}

export default function CompletePage({ loaderData }: Route.ComponentProps) {
  return (
    <div className="w-full max-w-sm space-y-4 text-center">
      <div className="space-y-3">
        <div className="text-5xl">✓</div>
        <h1 className="text-2xl font-bold">ข้อมูลถูกบันทึกแล้ว</h1>
        <p className="text-gray-300 leading-6">
          การบันทึกเสียงของคุณถูกบันทึกเรียบร้อยแล้ว
          <br />
          <br />
          ขอความร่วมมือผู้ใช้งานตอบแบบสอบถามเพื่อประเมินการใช้งานระบบสำหรับการพัฒนาต่อไป
        </p>
      </div>
      <a
        href={loaderData.surveyFormUrl}
        className="flex min-h-12 w-full items-center justify-center rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
      >
        ตอบแบบสอบถาม
      </a>
      <Link
        to="/"
        className="flex min-h-12 w-full items-center justify-center rounded-xl border border-gray-700 px-6 py-3 text-base font-semibold text-gray-400 transition hover:border-gray-500 hover:text-white active:scale-95"
      >
        บันทึกเสียงเพิ่ม
      </Link>
    </div>
  );
}
