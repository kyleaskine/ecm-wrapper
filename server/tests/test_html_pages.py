"""
Smoke tests that every HTML endpoint actually renders.

Nothing else in the suite renders a template: the admin tests hit JSON
endpoints, so a broken Jinja2Templates call surfaced only in production.
That is exactly what the starlette 1.x upgrade would have shipped --
TemplateResponse's first positional parameter became `request`, so every
old-style `TemplateResponse("name", {"request": request, ...})` call
raised at runtime while the suite stayed green.

Covers both rendering paths:
- TemplateResponse pages (the 15 call sites the upgrade migrated)
- endpoints that render via `templates.env.get_template(...).render(...)`
  and wrap the result in an HTMLResponse (the attempts fragments)

These assert that a page comes back as HTML, not what is in it -- the
point is to catch template wiring and rendering breakage, which is the
part dependency upgrades move.

Note on admin pages: failing admin auth does NOT return 4xx for HTML
routes. AdminAuthRedirect is handled into a 200 text/html shim, so
status-and-content-type alone cannot tell a rendered page from an auth
failure. assert_html_ok rejects the shim explicitly, and
test_admin_page_without_auth_renders_shim pins the shim's marker so that
rejection keeps meaning something.
"""
import pytest

from conftest import create_composite, get_test_engine

from app.models.attempts import ECMAttempt


COMPOSITE = "1234567890123456789012345678901234567891"

# Markers from get_unauthorized_redirect_html(); see module docstring.
SHIM_MARKERS = ("Redirecting to login", "<title>Unauthorized</title>")

PUBLIC_PAGES = [
    "/api/v1/dashboard/",
    "/api/v1/dashboard/testing-status",
    "/api/v1/dashboard/p1-testing-status",
    "/api/v1/dashboard/residue-status",
    "/api/v1/dashboard/curves",
    "/api/v1/dashboard/factors",
    "/api/v1/dashboard/leaderboard",
]

ADMIN_PAGES = [
    "/api/v1/admin/login",
    "/api/v1/admin/dashboard",
    "/api/v1/admin/inactive-composites",
    "/api/v1/admin/outstanding-work",
    "/api/v1/admin/recent-composites",
    "/api/v1/admin/residue-status",
]


def create_attempt(composite_id: int) -> int:
    """One ECM attempt, so the fragment endpoints have a row to render."""
    _, TestingSessionLocal = get_test_engine()
    db = TestingSessionLocal()
    try:
        attempt = ECMAttempt(
            composite_id=composite_id,
            client_id="test-client",
            method="ecm",
            b1=50000,
            b2=5_000_000,
            parametrization=1,
            curves_requested=100,
            curves_completed=100,
            program="gmp-ecm",
        )
        db.add(attempt)
        db.commit()
        return attempt.id
    finally:
        db.close()


def assert_html_ok(response, url):
    assert response.status_code == 200, (
        f"{url} returned {response.status_code}: {response.text[:500]}"
    )
    assert response.headers["content-type"].startswith("text/html"), url
    assert response.text.strip(), f"{url} rendered an empty body"
    for marker in SHIM_MARKERS:
        assert marker not in response.text, (
            f"{url} returned the admin auth-redirect shim, not a rendered "
            f"page (matched {marker!r})"
        )


class TestPublicPagesRender:
    @pytest.mark.parametrize("url", PUBLIC_PAGES)
    def test_page_renders(self, client, url):
        create_composite(COMPOSITE)
        assert_html_ok(client.get(url), url)

    def test_composite_details_renders(self, client):
        composite = create_composite(COMPOSITE)
        url = f"/api/v1/dashboard/composites/{composite['id']}/details"
        assert_html_ok(client.get(url), url)

    def test_attempts_fragment_renders(self, client):
        """Rendered via templates.env.get_template(...), not TemplateResponse."""
        composite = create_composite(COMPOSITE)
        create_attempt(composite["id"])
        url = f"/api/v1/dashboard/composites/{composite['id']}/attempts-fragment"
        assert_html_ok(client.get(url), url)


class TestAdminPagesRender:
    @pytest.mark.parametrize("url", ADMIN_PAGES)
    def test_page_renders(self, admin_client, url):
        create_composite(COMPOSITE)
        assert_html_ok(admin_client.get(url), url)

    def test_composite_details_renders(self, admin_client):
        composite = create_composite(COMPOSITE)
        url = f"/api/v1/admin/composites/{composite['id']}/details"
        assert_html_ok(admin_client.get(url), url)

    def test_attempts_fragment_renders(self, admin_client):
        """Rendered via templates.env.get_template(...), not TemplateResponse."""
        composite = create_composite(COMPOSITE)
        create_attempt(composite["id"])
        url = f"/api/v1/admin/composites/{composite['id']}/attempts-fragment"
        assert_html_ok(admin_client.get(url), url)


class TestAdminAuthShimIsDistinguishable:
    """Without this, every admin assertion above could pass on the shim."""

    def test_admin_page_without_auth_renders_shim(self, client):
        """Unauthenticated admin HTML returns 200 + the shim, not a 401.

        This is what makes assert_html_ok's shim rejection meaningful: it
        pins the marker the rejection looks for. If the shim's wording
        changes, this fails and SHIM_MARKERS must be updated with it.
        """
        response = client.get("/api/v1/admin/dashboard")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert any(marker in response.text for marker in SHIM_MARKERS)

    def test_shim_fails_assert_html_ok(self, client):
        """The guard actually rejects a shim response."""
        response = client.get("/api/v1/admin/dashboard")

        with pytest.raises(AssertionError, match="auth-redirect shim"):
            assert_html_ok(response, "/api/v1/admin/dashboard")
