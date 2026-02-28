import { Link } from "react-router";

export default function App() {
  return (
    <div className="w-full max-w-sm space-y-8 flex flex-col justify-center text-center">
      <div className="space-y-2">
        <div className="text-5xl">🩺</div>
        <h1 className="text-2xl font-bold">CoughSense</h1>
        <p className="pb-2 text-gray-400 text-sm">
          An AI-Based System for Cough Detection
        </p>
        <p>
          ระบบตรวจจับเสียงไอด้วย AI เพียงกดบันทึกเสียงและไอ
          ระบบจะวิเคราะห์และแสดงผลทันทีว่าเสียงที่บันทึก เป็นเสียงไอหรือไม่
        </p>
      </div>
      <Link
        to="/record"
        className="block min-h-12 w-full rounded-xl bg-blue-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500 active:scale-95"
      >
        เริ่มใช้งาน
      </Link>
      <div className="space-y-4">
        <p className="text-sm text-gray-400 leading-6">
          ชิ้นงานนี้เป็นส่วนหนึ่งของรายวิชา Team Project 2 (90641005)
          <br />
          คณะเทคโนโลยีสารสนเทศ สาขาเทคโนโลยีปัญญาประดิษฐ์
          <br />
          สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง
        </p>
        <p className="text-sm font-bold text-gray-300">
          Made with 💖 by AiCraft
        </p>
      </div>
    </div>
  );
}
