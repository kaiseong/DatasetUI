"use client";

import { useEffect } from "react";

/**
 * Warn before leaving with an unsaved object edit (or while one is saving):
 * on page unload, and on clicks/selects outside the editor, which is marked
 * with `data-object-editor` (the dataset/camera pickers live outside it).
 */
export function useUnsavedEditGuard(dirty: boolean, saving: boolean) {
  useEffect(() => {
    const warn = (event: BeforeUnloadEvent) => {
      if (dirty || saving) {
        event.preventDefault();
        event.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    const guard = (event: Event) => {
      const element = event.target instanceof Element ? event.target : null;
      if (
        (!dirty && !saving) ||
        !element ||
        element.closest("[data-object-editor]")
      )
        return;
      if (
        (event.type === "change" && element.matches("select")) ||
        element.closest("a,button")
      ) {
        if (
          saving ||
          !window.confirm("저장하지 않은 객체 편집을 버리고 이동할까요?")
        ) {
          event.preventDefault();
          event.stopImmediatePropagation();
        }
      }
    };
    document.addEventListener("change", guard, true);
    document.addEventListener("click", guard, true);
    return () => {
      window.removeEventListener("beforeunload", warn);
      document.removeEventListener("change", guard, true);
      document.removeEventListener("click", guard, true);
    };
  }, [dirty, saving]);
}
