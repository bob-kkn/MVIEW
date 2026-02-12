import unittest
from unittest.mock import patch

import app as m


class AppPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.client = m.app.test_client()
        m.dataset_list_cache["sig"] = None
        m.dataset_list_cache["datasets"] = []
        m._set_active([], [], use_tiles=True)

    def test_api_points_filters_with_bbox_and_limit(self):
        points = [
            {"lat": 37.1, "lon": 127.1, "track_seg_point_id": "a"},
            {"coordinates": [37.2, 127.2], "track_seg_point_id": "b"},
            {"lat": 38.0, "lon": 128.0, "track_seg_point_id": "c"},
        ]
        m._set_active(points, ["D1"], use_tiles=True)

        res = self.client.get("/api/points?bbox=127.0,37.0,127.3,37.3&limit=1")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["track_seg_point_id"], "a")

    def test_get_shapefile_datasets_uses_cache_when_signature_same(self):
        point_file = "/tmp/waypoint/D1.shp"

        with patch("app.glob.glob", return_value=[point_file]), \
             patch("app.get_image_folder_path", return_value="/tmp/img/D1"), \
             patch("app.get_absolute_path", side_effect=lambda p: f"/tmp/{p}"), \
             patch("app.os.path.exists", return_value=True), \
             patch("app.os.path.isdir", return_value=True), \
             patch("app._safe_mtime", return_value=123.0), \
             patch("app._quick_has_any_image", return_value=(True, 10)):
            first = m.get_shapefile_datasets()

        self.assertEqual(len(first), 1)

        with patch("app.glob.glob", return_value=[point_file]), \
             patch("app.get_image_folder_path", return_value="/tmp/img/D1"), \
             patch("app.get_absolute_path", side_effect=lambda p: f"/tmp/{p}"), \
             patch("app.os.path.exists", return_value=True), \
             patch("app.os.path.isdir", return_value=True), \
             patch("app._safe_mtime", return_value=123.0), \
             patch("app._quick_has_any_image", side_effect=AssertionError("should not be called")):
            second = m.get_shapefile_datasets()

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
