"""Researcher profiles."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from datasetui.api.context import RouterContext
from datasetui.database import DuplicateProfileNameError, ProfileNotFoundError
from datasetui.models import Profile, ProfileCreate, ProfileUpdate


def register(router: APIRouter, ctx: RouterContext) -> None:
    database = ctx.database

    @router.get("/profiles", response_model=list[Profile])
    def list_profiles(include_archived: bool = False) -> list[dict[str, Any]]:
        return database.list_profiles(include_archived=include_archived)

    @router.post(
        "/profiles", response_model=Profile, status_code=status.HTTP_201_CREATED
    )
    def create_profile(payload: ProfileCreate) -> dict[str, Any]:
        try:
            return database.create_profile(payload.name)
        except DuplicateProfileNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A profile with this name already exists",
            ) from exc

    @router.get("/profiles/{profile_id}", response_model=Profile)
    def get_profile(profile_id: str) -> dict[str, Any]:
        try:
            return database.get_profile(profile_id)
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc

    @router.patch("/profiles/{profile_id}", response_model=Profile)
    def update_profile(profile_id: str, payload: ProfileUpdate) -> dict[str, Any]:
        try:
            return database.update_profile(
                profile_id,
                name=payload.name,
                archived=payload.archived,
            )
        except ProfileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except DuplicateProfileNameError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A profile with this name already exists",
            ) from exc
