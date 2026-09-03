export const PROFILE_STORAGE_KEY = "datasetui.v1.profile-id";

export function storedProfileId(): string | null {
  try {
    return window.localStorage.getItem(PROFILE_STORAGE_KEY);
  } catch {
    return null;
  }
}
