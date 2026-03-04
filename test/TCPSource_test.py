# Copyright (C) 2017  DESY, Notkestr. 85, D-22607 Hamburg
#
# lavue is an image viewing program for photon science imaging detectors.
# Its usual application is as a live viewer using hidra as data source.
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation in  version 2
# of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA  02110-1301, USA.
#
# Authors:
#     Jan Kotanski <jan.kotanski@desy.de>
#
import unittest
import os
import sys
import random
import struct
import binascii
import time
import json
import numpy as np

import lavuelib
import lavuelib.imageSource

try:
    from pyqtgraph import QtWidgets
except Exception:
    from pyqtgraph import QtGui as QtWidgets


#  Qt-application
app = None

# if 64-bit machine
IS64BIT = struct.calcsize("P") == 8

if sys.version_info > (3,):
    long = int

# Path
path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, os.path.abspath(path))


def _make_frame(header_dict, pixel_data):
    """Helper: build a raw TCP frame (JSON header line + binary payload).

    :param header_dict: JSON-serialisable header dictionary
    :param pixel_data: raw bytes payload
    :returns: bytearray containing the complete frame
    """
    hdr_line = json.dumps(header_dict).encode("utf-8") + b"\n"
    return bytearray(hdr_line) + bytearray(pixel_data)


# test fixture
class TCPSourceTest(unittest.TestCase):
    def __init__(self, methodName):
        unittest.TestCase.__init__(self, methodName)
        global app
        if app is None:
            app = QtWidgets.QApplication([])
        app.setOrganizationName("DESY")
        app.setApplicationName("LaVue: unittests")
        app.setOrganizationDomain("desy.de")
        app.setApplicationVersion(lavuelib.__version__)

        try:
            self.__seed = long(binascii.hexlify(os.urandom(16)), 16)
        except NotImplementedError:
            self.__seed = long(time.time() * 256)
        self.__rnd = random.Random(self.__seed)

    def setUp(self):
        print("\nsetting up...")
        print("SEED = %s" % self.__seed)

    def tearDown(self):
        print("tearing down ...")

    # -- setConfiguration tests --

    def test_setConfiguration(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("myhost:9876")

        self.assertEqual(src._configuration, "myhost:9876")
        self.assertEqual(src._TCPSource__host, "myhost")
        self.assertEqual(src._TCPSource__port, 9876)
        self.assertEqual(src._TCPSource__bindaddress, "tcp://myhost:9876")
        self.assertFalse(src._initiated)

    def test_setConfiguration_new_value_resets_initiated(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("host1:1111")
        src._initiated = True
        src.setConfiguration("host2:2222")

        self.assertEqual(src._TCPSource__host, "host2")
        self.assertEqual(src._TCPSource__port, 2222)
        self.assertFalse(src._initiated)

    def test_setConfiguration_same_value_keeps_initiated(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("host1:1111")
        src._initiated = True
        src.setConfiguration("host1:1111")

        self.assertTrue(src._initiated)

    def test_setConfiguration_no_port(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("hostonly")

        self.assertEqual(src._TCPSource__host, "hostonly")
        self.assertIsNone(src._TCPSource__port)
        self.assertIsNone(src._TCPSource__bindaddress)

    def test_setConfiguration_empty(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("")

        self.assertEqual(src._TCPSource__host, "")
        self.assertIsNone(src._TCPSource__bindaddress)

    def test_setConfiguration_invalid_port(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("host:notanumber")

        # invalid port should result in None host/port/bindaddress
        self.assertIsNone(src._TCPSource__host)
        self.assertIsNone(src._TCPSource__port)
        self.assertIsNone(src._TCPSource__bindaddress)

    # -- getDataSection tests --

    def test_getDataSection_empty_buffer(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_partial_header(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # inject partial header (no newline yet)
        src._TCPSource__recvbuf = bytearray(b'{"width": 4, "height": 3, "bitDepth": 16')

        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_valid_16bit(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 4, 3
        pixel_values = np.arange(width * height, dtype="<u2")
        raw_data = pixel_values.tobytes()

        header = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw_data),
            "frameNumber": 7,
            "timeAtFrame": 1234567890.5,
        }

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("testhost:5555")
        src._TCPSource__recvbuf = _make_frame(header, raw_data)

        img, name, meta = src.getDataSection()

        self.assertIsNotNone(img)
        expected = pixel_values.reshape((height, width))
        self.assertTrue(np.allclose(img, np.transpose(expected)))
        self.assertEqual(img.dtype, np.dtype("<u2"))

        # verify name format
        self.assertIn("tcp://testhost:5555", name)
        self.assertIn("7", name)
        self.assertIn("1234567890.5", name)

        # verify metadata
        meta_dict = json.loads(meta)
        self.assertEqual(meta_dict["width"], width)
        self.assertEqual(meta_dict["height"], height)
        self.assertEqual(meta_dict["bitDepth"], 16)
        self.assertEqual(meta_dict["frameNumber"], 7)

    def test_getDataSection_valid_32bit(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 3, 2
        pixel_values = np.arange(width * height, dtype="<u4")
        raw_data = pixel_values.tobytes()

        header = {
            "width": width,
            "height": height,
            "bitDepth": 32,
            "dataSize": len(raw_data),
            "frameNumber": 99,
            "timeAtFrame": 9999.0,
        }

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("host32:7777")
        src._TCPSource__recvbuf = _make_frame(header, raw_data)

        img, name, meta = src.getDataSection()

        self.assertIsNotNone(img)
        expected = pixel_values.reshape((height, width))
        self.assertTrue(np.allclose(img, np.transpose(expected)))
        self.assertEqual(img.dtype, np.dtype("<u4"))

    def test_getDataSection_invalid_header_malformed_json(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # malformed JSON line followed by nothing else
        src._TCPSource__recvbuf = bytearray(b"this is not json\n")

        img, name, meta = src.getDataSection()

        # malformed header is skipped; no more data available
        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_invalid_header_missing_fields(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # header missing 'height' and 'dataSize'
        bad_header = json.dumps({"width": 4, "bitDepth": 16}).encode("utf-8")
        src._TCPSource__recvbuf = bytearray(bad_header + b"\n")

        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_invalid_header_bad_bitdepth(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # bitDepth=8 is not in (16, 32), should be ignored
        bad_header = json.dumps(
            {"width": 4, "height": 3, "bitDepth": 8, "dataSize": 24}
        ).encode("utf-8")
        src._TCPSource__recvbuf = bytearray(bad_header + b"\n")

        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_invalid_header_zero_datasize(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        bad_header = json.dumps(
            {"width": 4, "height": 3, "bitDepth": 16, "dataSize": 0}
        ).encode("utf-8")
        src._TCPSource__recvbuf = bytearray(bad_header + b"\n")

        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_partial_payload(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 4, 3
        pixel_values = np.arange(width * height, dtype="<u2")
        raw_data = pixel_values.tobytes()

        header = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw_data),
        }

        # build the full frame but only inject the header + partial payload
        hdr_line = json.dumps(header).encode("utf-8") + b"\n"
        half = len(raw_data) // 2

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("testhost:5555")
        src._TCPSource__recvbuf = bytearray(hdr_line + raw_data[:half])

        # first call: header parsed, but payload incomplete
        img, name, meta = src.getDataSection()
        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

        # internal state should now have expected_size set
        self.assertIsNotNone(src._TCPSource__expected_size)
        self.assertEqual(src._TCPSource__expected_size, len(raw_data))

        # now inject the rest of the payload
        src._TCPSource__recvbuf.extend(raw_data[half:])

        # NOTE: the second call hits the bit_depth scoping bug
        # (see test_getDataSection_bit_depth_bug). When the while loop
        # was already executed in a previous call and __expected_size
        # is set, bit_depth is undefined. We assert the bug here.
        with self.assertRaises(UnboundLocalError):
            src.getDataSection()

    def test_getDataSection_multiple_frames(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 2, 2

        # frame 1: 16-bit
        pix1 = np.array([10, 20, 30, 40], dtype="<u2")
        raw1 = pix1.tobytes()
        hdr1 = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw1),
            "frameNumber": 1,
        }

        # frame 2: 16-bit, different data
        pix2 = np.array([50, 60, 70, 80], dtype="<u2")
        raw2 = pix2.tobytes()
        hdr2 = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw2),
            "frameNumber": 2,
        }

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("multihost:6666")
        src._TCPSource__recvbuf = _make_frame(hdr1, raw1) + _make_frame(hdr2, raw2)

        # extract first frame
        img1, name1, meta1 = src.getDataSection()
        self.assertIsNotNone(img1)
        expected1 = pix1.reshape((height, width))
        self.assertTrue(np.allclose(img1, np.transpose(expected1)))
        meta1_dict = json.loads(meta1)
        self.assertEqual(meta1_dict["frameNumber"], 1)

        # extract second frame
        img2, name2, meta2 = src.getDataSection()
        self.assertIsNotNone(img2)
        expected2 = pix2.reshape((height, width))
        self.assertTrue(np.allclose(img2, np.transpose(expected2)))
        meta2_dict = json.loads(meta2)
        self.assertEqual(meta2_dict["frameNumber"], 2)

        # no more frames
        img3, name3, meta3 = src.getDataSection()
        self.assertIsNone(img3)
        self.assertIsNone(name3)
        self.assertIsNone(meta3)

    def test_getDataSection_empty_lines_skipped(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 2, 2
        pix = np.array([1, 2, 3, 4], dtype="<u2")
        raw = pix.tobytes()
        hdr = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw),
        }

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("emptylines:8888")
        # inject empty lines before the header
        hdr_line = json.dumps(hdr).encode("utf-8") + b"\n"
        src._TCPSource__recvbuf = bytearray(b"\n\n\n" + hdr_line + raw)

        img, name, meta = src.getDataSection()

        self.assertIsNotNone(img)
        expected = pix.reshape((height, width))
        self.assertTrue(np.allclose(img, np.transpose(expected)))

    def test_getDataSection_non_dict_json_skipped(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # a JSON array (not a dict) should be skipped
        src._TCPSource__recvbuf = bytearray(b"[1, 2, 3]\n")

        img, name, meta = src.getDataSection()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getDataSection_bit_depth_bug(self):
        """Expose the bit_depth scoping bug in getDataSection.

        When getDataSection is called with a header that sets
        __expected_size but the payload is not yet complete, the local
        variable 'bit_depth' is set inside the while loop. On the
        next call, the while loop is skipped (since __expected_size is
        already set), so 'bit_depth' is never assigned, leading to an
        UnboundLocalError at line 1888.
        """
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        width, height = 2, 2
        pix = np.arange(width * height, dtype="<u2")
        raw = pix.tobytes()

        header = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw),
        }

        hdr_line = json.dumps(header).encode("utf-8") + b"\n"

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("bughost:9999")

        # Call 1: inject only the header line, no payload bytes
        src._TCPSource__recvbuf = bytearray(hdr_line)
        img, name, meta = src.getDataSection()

        # Header was parsed, but no payload available
        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)
        self.assertIsNotNone(src._TCPSource__expected_size)

        # Call 2: inject the payload bytes
        src._TCPSource__recvbuf.extend(raw)

        # This call should raise UnboundLocalError because the
        # while loop is skipped and 'bit_depth' was never assigned
        # in this invocation scope.
        with self.assertRaises(UnboundLocalError):
            src.getDataSection()

    # -- __make_name tests --

    def test_make_name_with_all_fields(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("namehost:1234")

        header = {"frameNumber": 42, "timeAtFrame": 1700000000.123}
        name = src._TCPSource__make_name(header)

        self.assertIn("tcp://namehost:1234", name)
        self.assertIn("42", name)
        self.assertIn("1700000000.123", name)

    def test_make_name_missing_fields(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        src.setConfiguration("namehost:1234")

        # no frameNumber or timeAtFrame
        header = {"width": 10}
        name = src._TCPSource__make_name(header)

        self.assertIn("tcp://namehost:1234", name)
        # empty strings for missing fields
        self.assertEqual(name, "tcp://namehost:1234  ()")

    def test_make_name_no_bindaddress(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # no configuration set -> no bindaddress

        header = {"frameNumber": 1, "timeAtFrame": 0.0}
        name = src._TCPSource__make_name(header)

        self.assertEqual(name, " 1 (0.0)")

    # -- connect / disconnect tests --

    def test_connect_no_host(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # no configuration => connect should fail
        result = src.connect()

        self.assertFalse(result)

    def test_disconnect_when_not_connected(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()
        # disconnect should not raise even if never connected
        src.disconnect()
        self.assertIsNone(src._TCPSource__socket)

    def test_connect_disconnect_with_real_server(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        import socket
        import threading

        # start a simple TCP echo server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        accepted = [None]

        def accept_one():
            accepted[0], _ = server.accept()

        t = threading.Thread(target=accept_one)
        t.daemon = True
        t.start()

        try:
            src = lavuelib.imageSource.TCPSource(timeout=5000)
            src.setConfiguration("127.0.0.1:%d" % port)
            result = src.connect()

            self.assertTrue(result)
            self.assertIsNotNone(src._TCPSource__socket)
            self.assertTrue(src._initiated)

            src.disconnect()
            self.assertIsNone(src._TCPSource__socket)
        finally:
            if accepted[0]:
                accepted[0].close()
            server.close()
            t.join(timeout=2)

    # -- getData tests --

    def test_getData_not_connected_no_config(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        src = lavuelib.imageSource.TCPSource()

        img, name, meta = src.getData()

        self.assertIsNone(img)
        self.assertIsNone(name)
        self.assertIsNone(meta)

    def test_getData_with_real_server(self):
        fun = sys._getframe().f_code.co_name
        print("Run: %s.%s() " % (self.__class__.__name__, fun))

        import socket
        import threading
        import select

        width, height = 3, 2
        pix = np.arange(width * height, dtype="<u2")
        raw = pix.tobytes()
        header = {
            "width": width,
            "height": height,
            "bitDepth": 16,
            "dataSize": len(raw),
            "frameNumber": 1,
        }
        frame = bytes(_make_frame(header, raw))

        # start a simple TCP server that sends a frame
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen(1)

        def serve():
            conn, _ = server.accept()
            conn.sendall(frame)
            # keep connection open briefly so client can read
            time.sleep(0.5)
            conn.close()

        t = threading.Thread(target=serve)
        t.daemon = True
        t.start()

        try:
            src = lavuelib.imageSource.TCPSource(timeout=5000)
            src.setConfiguration("127.0.0.1:%d" % port)
            src.connect()

            # wait for the non-blocking connect to complete and data to arrive
            time.sleep(0.3)

            img, name, meta = src.getData()

            self.assertIsNotNone(img)
            expected = pix.reshape((height, width))
            self.assertTrue(np.allclose(img, np.transpose(expected)))
            self.assertEqual(img.dtype, np.dtype("<u2"))

            meta_dict = json.loads(meta)
            self.assertEqual(meta_dict["frameNumber"], 1)

            src.disconnect()
        finally:
            server.close()
            t.join(timeout=2)


if __name__ == "__main__":
    if app is None:
        app = QtWidgets.QApplication([])
    unittest.main()
