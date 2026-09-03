"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import { LuUserRound, LuX } from "react-icons/lu";
import {
  createProfile as createProfileRequest,
  listProfiles,
  type Profile,
} from "@/lib/workbench-api";

const PROFILE_STORAGE_KEY = "datasetui.v1.profile-id";

type ProfileContextValue = {
  profiles: Profile[];
  currentProfile: Profile | null;
  loading: boolean;
  openProfileDialog: () => void;
  selectProfile: (profile: Profile) => void;
  refreshProfiles: () => Promise<void>;
};

const ProfileContext = createContext<ProfileContextValue | null>(null);

export function useProfile() {
  const value = useContext(ProfileContext);
  if (!value) throw new Error("useProfile must be used inside ProfileProvider");
  return value;
}

export function ProfileProvider({ children }: { children: ReactNode }) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [currentProfile, setCurrentProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const refreshProfiles = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const loaded = await listProfiles();
      setProfiles(loaded);
      const storedId = window.localStorage.getItem(PROFILE_STORAGE_KEY);
      const selected =
        loaded.find((profile) => profile.id === storedId) ?? null;
      setCurrentProfile(selected);
      if (!selected) setDialogOpen(true);
    } catch (requestError) {
      setCurrentProfile(null);
      setLoadError(
        requestError instanceof Error
          ? requestError.message
          : "사용자 목록을 불러오지 못했습니다.",
      );
      setDialogOpen(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshProfiles();
  }, [refreshProfiles]);

  const selectProfile = useCallback((profile: Profile) => {
    window.localStorage.setItem(PROFILE_STORAGE_KEY, profile.id);
    setCurrentProfile(profile);
    setDialogOpen(false);
  }, []);

  const value = useMemo(
    () => ({
      profiles,
      currentProfile,
      loading,
      openProfileDialog: () => setDialogOpen(true),
      selectProfile,
      refreshProfiles,
    }),
    [profiles, currentProfile, loading, selectProfile, refreshProfiles],
  );

  return (
    <ProfileContext.Provider value={value}>
      {children}
      <ProfileDialog
        open={dialogOpen}
        profiles={profiles}
        currentProfile={currentProfile}
        canClose={Boolean(currentProfile)}
        loadError={loadError}
        retrying={loading}
        onClose={() => currentProfile && setDialogOpen(false)}
        onRetry={refreshProfiles}
        onSelect={selectProfile}
        onCreated={(profile) => {
          setProfiles((items) =>
            [...items, profile].sort((a, b) => a.name.localeCompare(b.name)),
          );
          selectProfile(profile);
        }}
      />
    </ProfileContext.Provider>
  );
}

function ProfileDialog({
  open,
  profiles,
  currentProfile,
  canClose,
  loadError,
  retrying,
  onClose,
  onRetry,
  onSelect,
  onCreated,
}: {
  open: boolean;
  profiles: Profile[];
  currentProfile: Profile | null;
  canClose: boolean;
  loadError: string | null;
  retrying: boolean;
  onClose: () => void;
  onRetry: () => Promise<void>;
  onSelect: (profile: Profile) => void;
  onCreated: (profile: Profile) => void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      dialog.showModal();
      window.setTimeout(() => inputRef.current?.focus(), 0);
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  async function handleCreate(event: FormEvent) {
    event.preventDefault();
    const normalized = name.trim();
    if (!normalized) return;
    setSubmitting(true);
    setError(null);
    try {
      const profile = await createProfileRequest(normalized);
      setName("");
      onCreated(profile);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "사용자를 만들지 못했습니다.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      className="profile-dialog"
      onCancel={(event) => {
        if (!canClose) event.preventDefault();
        else onClose();
      }}
      onClose={() => canClose && onClose()}
      aria-labelledby="profile-dialog-title"
    >
      <div className="profile-dialog__topline" />
      <div className="flex items-start justify-between gap-6">
        <div>
          <p className="workbench-eyebrow">작업자 선택</p>
          <h2 id="profile-dialog-title" className="mt-2 text-2xl font-semibold">
            누가 작업하나요?
          </h2>
          <p className="mt-2 max-w-md text-sm text-[var(--text-muted)]">
            작업 기록에 표시할 이름입니다. 비밀번호는 필요하지 않습니다.
          </p>
        </div>
        {canClose && (
          <button
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="사용자 선택 닫기"
          >
            <LuX aria-hidden />
          </button>
        )}
      </div>

      {loadError && (
        <div
          className="mt-5 rounded-md bg-red-400/8 px-3 py-2 text-sm text-red-300"
          role="alert"
        >
          <p>사용자 목록에 연결할 수 없습니다. {loadError}</p>
          <button
            type="button"
            className="workbench-button mt-3"
            onClick={() => void onRetry()}
            disabled={retrying}
          >
            {retrying ? "다시 연결 중…" : "다시 시도"}
          </button>
        </div>
      )}

      {profiles.length > 0 && (
        <div className="mt-7 grid gap-2 sm:grid-cols-2">
          {profiles.map((profile) => (
            <button
              type="button"
              key={profile.id}
              className={`profile-choice ${currentProfile?.id === profile.id ? "profile-choice--current" : ""}`}
              onClick={() => onSelect(profile)}
            >
              <span className="profile-choice__avatar">
                <LuUserRound aria-hidden />
              </span>
              <span className="min-w-0 truncate">{profile.name}</span>
            </button>
          ))}
        </div>
      )}

      <form
        onSubmit={handleCreate}
        className="mt-7 border-t border-white/8 pt-6"
      >
        <label htmlFor="new-profile-name" className="text-sm font-medium">
          새 사용자 추가
        </label>
        <div className="mt-2 flex gap-2">
          <input
            ref={inputRef}
            id="new-profile-name"
            value={name}
            onChange={(event) => setName(event.target.value)}
            className="workbench-input min-w-0 flex-1"
            placeholder="이름 입력"
            maxLength={80}
            autoComplete="off"
          />
          <button
            type="submit"
            className="workbench-button workbench-button--primary"
            disabled={submitting || !name.trim()}
          >
            {submitting ? "추가 중…" : "추가"}
          </button>
        </div>
        {error && (
          <p className="mt-2 text-sm text-red-300" role="alert">
            {error}
          </p>
        )}
      </form>
    </dialog>
  );
}
