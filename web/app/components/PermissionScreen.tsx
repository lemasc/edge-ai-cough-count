import { useState } from "react";
import type { PermissionStatus } from "../types";

type Props = {
  requestPermission: () => Promise<"granted" | "denied">;
  onBeginRecording?: () => void;
};

type BadgeStatus = "unknown" | "granted" | "denied" | "unavailable";

function StatusBadge({
  status,
  label,
}: {
  status: BadgeStatus;
  label: string;
}) {
  const colors: Record<BadgeStatus, string> = {
    unknown: "bg-yellow-500/20 text-yellow-300 border-yellow-500/40",
    granted: "bg-green-500/20 text-green-300 border-green-500/40",
    denied: "bg-red-500/20 text-red-300 border-red-500/40",
    unavailable: "bg-gray-500/20 text-gray-400 border-gray-500/40",
  };

  const icons: Record<BadgeStatus, string> = {
    unknown: "?",
    granted: "✓",
    denied: "✗",
    unavailable: "—",
  };

  const statusText: Record<BadgeStatus, string> = {
    unknown: "ไม่ทราบสถานะ",
    granted: "ได้รับอนุญาต",
    denied: "ถูกปฏิเสธ",
    unavailable: "ไม่สามารถใช้ได้",
  };

  return (
    <div
      className={`flex items-center justify-between rounded-lg border px-4 py-3 ${colors[status]}`}
    >
      <span className="text-sm font-medium">{label}</span>
      <span className="text-sm font-semibold">
        {statusText[status]} {icons[status]}
      </span>
    </div>
  );
}

export function PermissionScreen({
  requestPermission,
  onBeginRecording,
}: Props) {
  const [phase, setPhase] = useState<
    "idle" | "requesting-permissions" | "ready"
  >("idle");
  const [permissions, setPermissions] = useState<PermissionStatus>({
    audio: "unknown",
  });
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const showBeginRecording = phase === "ready" && Boolean(onBeginRecording);
  const showPermissions = phase !== "idle" || permissions.audio !== "unknown";

  const onRequestPermission = () => {
    setErrorMessage(null);

    if (!navigator.mediaDevices?.getUserMedia) {
      setPermissions({ audio: "unavailable" });
      setErrorMessage(
        "เบราวเซอร์ของคุณไม่รองรับการเข้าถึงไมโครโฟน กรุณาใช้เบราวเซอร์ Google Chrome หรือ Safari",
      );
      setPhase("idle");
      return;
    }

    setPhase("requesting-permissions");
    void requestPermission().then((result) => {
      if (result === "granted") {
        setPermissions({ audio: "granted" });
        setPhase("ready");
        return;
      }

      setPermissions({ audio: "denied" });
      setPhase("idle");
    });
  };

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="text-center space-y-2">
        <h1 className="text-2xl font-bold pb-1">ก่อนจะเริ่มต้น</h1>
        <p>กรุณาอ่านรายละเอียดการบันทึกเสียง ดังนี้:</p>
      </div>
      <div className="bg-white/10 p-4 rounded-lg">
        <ol className="list-decimal list-outside pl-6 leading-7">
          <li>ผู้ใช้งานต้องบันทึกเสียงไอ ภายในเวลาไม่เกิน 10 วินาที</li>
          <li>
            ระหว่างบันทึกเสียง ให้{" "}
            <b className="text-blue-300">
              ถือโทรศัพท์กลับหัว โดยให้ไมโครโฟนอยู่ด้านบน
            </b>{" "}
            และ <b className="text-blue-300">แนบโทรศัพท์ไว้ที่กึ่งกลางหน้าอก</b>
          </li>
          <li>
            หลังจากครบระยะเวลาแล้ว
            เสียงของคุณจะถูกนำไปประมวลผลเพื่อตรวจจับและนับจำนวนการไอทั้งหมด
          </li>
          <li>
            ผู้ใช้งานสามารถอัดเสียงในสภาพแวดล้อมที่มีเสียงรบกวนด้วยได้
            เพื่อทดสอบความสามารถในการตรวจจับเสียงไอในสภาพแวดล้อมต่าง ๆ
          </li>
        </ol>
      </div>

      {showPermissions && (
        <div className="space-y-2">
          <p className="text-xs font-semibold tracking-wider text-gray-500">
            สิทธิ์การเข้าถึง
          </p>
          <StatusBadge status={permissions.audio} label="ไมโครโฟน" />
          {errorMessage && (
            <p className="text-xs text-red-400 leading-5">{errorMessage}</p>
          )}
          {!errorMessage && permissions.audio === "denied" && (
            <p className="text-xs text-red-400 leading-5">
              สิทธิ์การเข้าถึงไมโครโฟนถูกปฏิเสธ
              กรุณาอนุญาตการเข้าถึงในเบราวเซอร์ และ Refresh หน้าเว็บใหม่อีกครั้ง
            </p>
          )}
        </div>
      )}

      {!showBeginRecording ? (
        <>
          <p className="text-center font-bold">
            เพื่อเริ่มต้น กรุณาอนุญาตสิทธิ์การเข้าถึงไมโครโฟน
          </p>
          <button
            type="button"
            onClick={onRequestPermission}
            disabled={phase === "requesting-permissions"}
            className="min-h-12 w-full rounded-xl bg-blue-600 disabled:bg-blue-600/50 px-6 py-3 text-base font-semibold text-white transition hover:bg-blue-500"
          >
            {phase === "requesting-permissions"
              ? "กำลังขอสิทธิ์เข้าถึง..."
              : "ขอสิทธิ์เข้าถึงไมโครโฟน"}
          </button>
        </>
      ) : (
        <>
          <p className="text-center font-bold">
            เมื่อพร้อม ให้เริ่มต้นการบันทึก โดยระบบจะนับถอยหลัง 3 วินาที
          </p>
          <button
            type="button"
            onClick={onBeginRecording}
            disabled={permissions.audio !== "granted"}
            className="min-h-12 w-full rounded-xl bg-green-600 px-6 py-3 text-base font-semibold text-white transition hover:bg-green-500 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
          >
            เริ่มต้นการบันทึก
          </button>
        </>
      )}
    </div>
  );
}
