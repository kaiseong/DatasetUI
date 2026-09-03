"use client";

import {
  LuCheck,
  LuCircleUserRound,
  LuUserPlus,
  LuUsersRound,
} from "react-icons/lu";
import { useProfile } from "@/components/workbench/profile-context";

export default function ProfilesPage() {
  const {
    profiles,
    currentProfile,
    loading,
    selectProfile,
    openProfileDialog,
  } = useProfile();

  return (
    <div className="workbench-page">
      <section className="workbench-page__heading">
        <div>
          <p className="workbench-eyebrow">PEOPLE</p>
          <h1>사용자</h1>
          <p>로그인 계정이 아니라 작업 기록에 표시되는 팀원 이름입니다.</p>
        </div>
        <button
          type="button"
          className="workbench-button workbench-button--primary"
          onClick={openProfileDialog}
        >
          <LuUserPlus aria-hidden /> 사용자 추가
        </button>
      </section>

      <aside className="profile-explainer">
        <LuCircleUserRound aria-hidden />
        <div>
          <strong>비밀번호 없이 선택합니다</strong>
          <p>
            같은 연구실 네트워크에서 누가 Flag·Trim·Export를 했는지 구분하기
            위한 이름입니다.
          </p>
        </div>
      </aside>

      {loading ? (
        <div className="profile-grid" aria-label="사용자 불러오는 중">
          {[0, 1, 2].map((item) => (
            <div key={item} className="profile-card profile-card--skeleton" />
          ))}
        </div>
      ) : (
        <section className="profile-grid" aria-label="사용자 목록">
          {profiles.map((profile) => {
            const current = currentProfile?.id === profile.id;
            return (
              <button
                key={profile.id}
                type="button"
                className={`profile-card ${current ? "profile-card--current" : ""}`}
                onClick={() => selectProfile(profile)}
              >
                <span className="profile-card__avatar">
                  <LuCircleUserRound aria-hidden />
                </span>
                <span>
                  <strong>{profile.name}</strong>
                  <small>
                    {current
                      ? "현재 사용자"
                      : `${formatDate(profile.created_at)} 추가`}
                  </small>
                </span>
                {current && (
                  <LuCheck className="profile-card__check" aria-hidden />
                )}
              </button>
            );
          })}
          <button
            type="button"
            className="profile-card profile-card--add"
            onClick={openProfileDialog}
          >
            <span className="profile-card__avatar">
              <LuUsersRound aria-hidden />
            </span>
            <span>
              <strong>새 사용자</strong>
              <small>팀원 이름 추가</small>
            </span>
          </button>
        </section>
      )}
    </div>
  );
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("ko-KR", {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(value));
}
