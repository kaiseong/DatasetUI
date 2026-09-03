"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  getEpisodeFlags,
  updateEpisodeFlags,
  WorkbenchApiError,
  type EpisodeFlagChange,
  type EpisodeFlags,
} from "@/lib/workbench-api";
import { storedProfileId } from "@/lib/profile-selection";

const SESSION_STORAGE_PREFIX = "flagged-episodes";

type FlaggedEpisodesContextType = {
  flagged: Set<number>;
  count: number;
  has: (id: number) => boolean;
  toggle: (id: number) => void;
  addMany: (ids: number[]) => void;
  clear: () => void;
  persistent: boolean;
  loading: boolean;
  syncing: boolean;
  error: string | null;
  profileRequired: boolean;
};

const FlaggedEpisodesContext = createContext<
  FlaggedEpisodesContextType | undefined
>(undefined);

export function useFlaggedEpisodes() {
  const context = useContext(FlaggedEpisodesContext);
  if (!context) {
    throw new Error(
      "useFlaggedEpisodes must be used within FlaggedEpisodesProvider",
    );
  }
  return context;
}

function sessionKey(scope: string) {
  return `${SESSION_STORAGE_PREFIX}:${scope}`;
}

function validEpisode(id: number, totalEpisodes?: number) {
  return (
    Number.isInteger(id) &&
    id >= 0 &&
    (totalEpisodes === undefined || id < totalEpisodes)
  );
}

export const FlaggedEpisodesProvider: React.FC<{
  children: React.ReactNode;
  datasetId?: string;
  storageScope?: string;
  totalEpisodes?: number;
}> = ({ children, datasetId, storageScope = "viewer", totalEpisodes }) => {
  const [flagged, setFlagged] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profileRequired, setProfileRequired] = useState(false);
  const flaggedRef = useRef(flagged);
  const profileIdRef = useRef<string | null>(null);
  const revisionRef = useRef(0);
  const mutationQueueRef = useRef(Promise.resolve());
  const pendingMutationsRef = useRef(0);

  const replaceFlags = useCallback((indices: number[]) => {
    const next = new Set(indices);
    flaggedRef.current = next;
    setFlagged(next);
  }, []);

  const applyServerFlags = useCallback(
    (response: EpisodeFlags) => {
      revisionRef.current = response.revision;
      replaceFlags(response.episode_indices);
    },
    [replaceFlags],
  );

  const loadPersistentFlags = useCallback(async () => {
    if (!datasetId) return;
    const profileId = storedProfileId();
    profileIdRef.current = profileId;
    if (!profileId) {
      revisionRef.current = 0;
      replaceFlags([]);
      setProfileRequired(true);
      setError("라이브러리에서 작업자를 먼저 선택해 주세요.");
      setLoading(false);
      return;
    }
    setProfileRequired(false);
    try {
      applyServerFlags(await getEpisodeFlags(datasetId, profileId));
      setError(null);
    } catch (requestError) {
      replaceFlags([]);
      if (
        requestError instanceof WorkbenchApiError &&
        requestError.status === 404
      ) {
        setProfileRequired(true);
      }
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Flag를 불러오지 못했습니다.",
      );
    } finally {
      setLoading(false);
    }
  }, [applyServerFlags, datasetId, replaceFlags]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    setProfileRequired(false);
    revisionRef.current = 0;
    mutationQueueRef.current = Promise.resolve();
    if (datasetId) {
      void loadPersistentFlags();
      return;
    }

    try {
      const raw = sessionStorage.getItem(sessionKey(storageScope));
      replaceFlags(raw ? (JSON.parse(raw) as number[]) : []);
    } catch {
      replaceFlags([]);
    }
    setLoading(false);
  }, [datasetId, loadPersistentFlags, replaceFlags, storageScope]);

  useEffect(() => {
    if (datasetId || loading) return;
    try {
      sessionStorage.setItem(
        sessionKey(storageScope),
        JSON.stringify([...flagged]),
      );
    } catch {
      // A private browser session may disable storage; flags still work in memory.
    }
  }, [datasetId, flagged, loading, storageScope]);

  const persistChanges = useCallback(
    (changes: EpisodeFlagChange[]) => {
      if (!datasetId || !changes.length) return;
      const profileId = profileIdRef.current;
      if (!profileId) {
        setProfileRequired(true);
        setError("라이브러리에서 작업자를 먼저 선택해 주세요.");
        return;
      }
      pendingMutationsRef.current += 1;
      setSyncing(true);

      const run = async () => {
        try {
          let response: EpisodeFlags;
          try {
            response = await updateEpisodeFlags(
              datasetId,
              profileId,
              revisionRef.current,
              changes,
            );
          } catch (requestError) {
            if (
              !(requestError instanceof WorkbenchApiError) ||
              requestError.status !== 409
            ) {
              throw requestError;
            }
            const latest = await getEpisodeFlags(datasetId, profileId);
            revisionRef.current = latest.revision;
            response = await updateEpisodeFlags(
              datasetId,
              profileId,
              latest.revision,
              changes,
            );
          }
          applyServerFlags(response);
          setError(null);
        } catch (requestError) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "Flag를 저장하지 못했습니다.",
          );
          await loadPersistentFlags();
        } finally {
          pendingMutationsRef.current -= 1;
          if (pendingMutationsRef.current === 0) setSyncing(false);
        }
      };
      mutationQueueRef.current = mutationQueueRef.current.then(run, run);
    },
    [applyServerFlags, datasetId, loadPersistentFlags],
  );

  const applyOptimisticChanges = useCallback(
    (changes: EpisodeFlagChange[]) => {
      if (!changes.length) return;
      if (datasetId && !profileIdRef.current) {
        setProfileRequired(true);
        setError("라이브러리에서 작업자를 먼저 선택해 주세요.");
        return;
      }
      const next = new Set(flaggedRef.current);
      for (const change of changes) {
        if (change.flagged) next.add(change.episode_index);
        else next.delete(change.episode_index);
      }
      flaggedRef.current = next;
      setFlagged(next);
      persistChanges(changes);
    },
    [datasetId, persistChanges],
  );

  const toggle = useCallback(
    (id: number) => {
      if (!validEpisode(id, totalEpisodes)) return;
      applyOptimisticChanges([
        { episode_index: id, flagged: !flaggedRef.current.has(id) },
      ]);
    },
    [applyOptimisticChanges, totalEpisodes],
  );

  const addMany = useCallback(
    (ids: number[]) => {
      const changes = [...new Set(ids)]
        .filter(
          (id) =>
            validEpisode(id, totalEpisodes) && !flaggedRef.current.has(id),
        )
        .map((id) => ({ episode_index: id, flagged: true }));
      applyOptimisticChanges(changes);
    },
    [applyOptimisticChanges, totalEpisodes],
  );

  const clear = useCallback(() => {
    applyOptimisticChanges(
      [...flaggedRef.current].map((id) => ({
        episode_index: id,
        flagged: false,
      })),
    );
  }, [applyOptimisticChanges]);

  const has = useCallback((id: number) => flagged.has(id), [flagged]);

  const value = useMemo(
    () => ({
      flagged,
      count: flagged.size,
      has,
      toggle,
      addMany,
      clear,
      persistent: Boolean(datasetId),
      loading,
      syncing,
      error,
      profileRequired,
    }),
    [
      addMany,
      clear,
      datasetId,
      error,
      flagged,
      has,
      loading,
      profileRequired,
      syncing,
      toggle,
    ],
  );

  return (
    <FlaggedEpisodesContext.Provider value={value}>
      {children}
    </FlaggedEpisodesContext.Provider>
  );
};
