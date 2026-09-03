"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import {
  LuActivity,
  LuArchive,
  LuCircleUserRound,
  LuDatabase,
  LuExternalLink,
  LuUsersRound,
  LuWifi,
  LuWifiOff,
  LuMerge,
  LuShieldCheck,
  LuSend,
} from "react-icons/lu";
import { getSystemHealth } from "@/lib/workbench-api";
import { ProfileProvider, useProfile } from "./profile-context";

const NAV_ITEMS = [
  { href: "/library", label: "라이브러리", icon: LuDatabase },
  { href: "/merge", label: "합치기", icon: LuMerge },
  { href: "/validate", label: "검증", icon: LuShieldCheck },
  { href: "/deliver", label: "전달", icon: LuSend },
  { href: "/jobs", label: "작업 기록", icon: LuActivity },
  { href: "/profiles", label: "사용자", icon: LuUsersRound },
];

export default function WorkbenchShell({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProfileProvider>
      <ShellFrame>{children}</ShellFrame>
    </ProfileProvider>
  );
}

function ShellFrame({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { currentProfile, loading, openProfileDialog } = useProfile();

  return (
    <div className="workbench-shell" lang="ko">
      <header className="workbench-header">
        <Link
          href="/library"
          className="workbench-brand"
          aria-label="DatasetUI 홈"
        >
          <span className="workbench-brand__mark" aria-hidden>
            <LuArchive />
          </span>
          <span>
            <strong>DatasetUI</strong>
            <small>SHARED LAB LEDGER</small>
          </span>
        </Link>

        <nav className="workbench-nav" aria-label="주요 메뉴">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={active ? "is-active" : ""}
                aria-current={active ? "page" : undefined}
              >
                <Icon aria-hidden />
                {item.label}
              </Link>
            );
          })}
        </nav>

        <div className="flex items-center gap-2">
          <SystemBadge />
          <Link href="/viewer" className="workbench-viewer-link">
            HF Viewer
            <LuExternalLink aria-hidden />
          </Link>
          <button
            type="button"
            className="profile-trigger"
            onClick={openProfileDialog}
            disabled={loading}
          >
            <LuCircleUserRound aria-hidden />
            <span>
              {loading ? "확인 중…" : (currentProfile?.name ?? "사용자 선택")}
            </span>
          </button>
        </div>
      </header>

      <main className="workbench-main">{children}</main>

      <nav className="workbench-mobile-nav" aria-label="모바일 주요 메뉴">
        {NAV_ITEMS.map((item) => {
          const active = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={active ? "is-active" : ""}
              aria-current={active ? "page" : undefined}
            >
              <Icon aria-hidden />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}

function SystemBadge() {
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    let timer: number | undefined;
    const check = async () => {
      try {
        const health = await getSystemHealth();
        if (active) setOnline(health.ok);
      } catch {
        if (active) setOnline(false);
      }
      if (active) timer = window.setTimeout(check, 30_000);
    };
    void check();
    return () => {
      active = false;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, []);

  return (
    <span
      className={`system-badge ${online === false ? "system-badge--offline" : ""}`}
      title={
        online === false ? "DatasetUI 서버 연결 끊김" : "DatasetUI 서버 연결됨"
      }
    >
      {online === false ? <LuWifiOff aria-hidden /> : <LuWifi aria-hidden />}
      <span>
        {online === null ? "확인 중" : online ? "연결됨" : "연결 끊김"}
      </span>
    </span>
  );
}
