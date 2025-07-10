#!/usr/bin/env python3
"""
Test WebSocket implementation with multiple concurrent connections.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid

import websockets
from websockets.client import WebSocketClientProtocol

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketTestClient:
    """Test client for WebSocket connections."""

    def __init__(
        self,
        url: str = "ws://localhost:8000/api/v1/ws",
        user_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.url = url
        self.user_id = user_id or f"test_user_{uuid.uuid4().hex[:8]}"
        self.token = token
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connection_id: Optional[str] = None
        self.messages_received: List[Dict[str, Any]] = []
        self.connected = False

    async def connect(self):
        """Connect to WebSocket server."""
        try:
            # Add token to URL if provided
            connect_url = self.url
            if self.token:
                connect_url = f"{self.url}?token={self.token}"

            self.websocket = await websockets.connect(connect_url)
            self.connected = True

            # Start message listener
            asyncio.create_task(self._listen_messages())

            logger.info(f"Client {self.user_id} connected to {self.url}")

            # Wait for connection confirmation
            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Connection failed for {self.user_id}: {e}")
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            logger.info(f"Client {self.user_id} disconnected")

    async def _listen_messages(self):
        """Listen for incoming messages."""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                self.messages_received.append(data)

                # Handle specific message types
                if data.get("type") == "connected":
                    self.connection_id = data["data"]["connection_id"]
                    logger.info(
                        f"Client {self.user_id} assigned connection ID: {self.connection_id}"
                    )

                elif data.get("type") == "generation_update":
                    logger.info(f"Client {self.user_id} received generation update: {data['data']}")

                elif data.get("type") == "chat_message":
                    logger.info(f"Client {self.user_id} received chat: {data['data']['message']}")

                elif data.get("type") == "error":
                    logger.error(f"Client {self.user_id} received error: {data.get('error')}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {self.user_id} connection closed")
            self.connected = False
        except Exception as e:
            logger.error(f"Listen error for {self.user_id}: {e}")
            self.connected = False

    async def send_message(self, msg_type: str, data: Dict[str, Any]):
        """Send a message to the server."""
        if not self.websocket or not self.connected:
            raise RuntimeError("Not connected")

        message = {
            "type": msg_type,
            "data": data,
        }

        await self.websocket.send(json.dumps(message))
        logger.info(f"Client {self.user_id} sent {msg_type}: {data}")

    async def ping(self):
        """Send ping message."""
        await self.send_message("ping", {})

    async def join_room(self, room_id: str):
        """Join a room."""
        await self.send_message("join_room", {"room_id": room_id})

    async def leave_room(self, room_id: str):
        """Leave a room."""
        await self.send_message("leave_room", {"room_id": room_id})

    async def send_chat(self, room_id: str, message: str):
        """Send chat message to a room."""
        await self.send_message(
            "chat_message",
            {
                "room_id": room_id,
                "message": message,
            },
        )

    async def start_generation(self, task_id: str, prompt: str):
        """Start generation task."""
        await self.send_message(
            "generation_start",
            {
                "task_id": task_id,
                "prompt": prompt,
            },
        )

    async def subscribe_channels(self, channels: List[str]):
        """Subscribe to channels."""
        await self.send_message("subscribe", {"channels": channels})


class WebSocketLoadTester:
    """Load tester for WebSocket connections."""

    def __init__(self, base_url: str = "ws://localhost:8000/api/v1/ws"):
        self.base_url = base_url
        self.clients: List[WebSocketTestClient] = []
        self.stats = {
            "connections_created": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None,
        }

    async def create_clients(self, count: int):
        """Create multiple test clients."""
        logger.info(f"Creating {count} WebSocket clients...")

        tasks = []
        for i in range(count):
            client = WebSocketTestClient(
                url=self.base_url,
                user_id=f"load_test_user_{i}",
            )
            self.clients.append(client)
            tasks.append(client.connect())

        # Connect all clients
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful connections
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Client {i} connection failed: {result}")
                self.stats["errors"] += 1
            else:
                self.stats["connections_created"] += 1

        logger.info(f"Created {self.stats['connections_created']} connections")

    async def test_ping_pong(self):
        """Test ping/pong with all clients."""
        logger.info("Testing ping/pong...")

        tasks = []
        for client in self.clients:
            if client.connected:
                tasks.append(client.ping())

        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats["messages_sent"] += len(tasks)

        # Wait for pongs
        await asyncio.sleep(1)

    async def test_room_operations(self):
        """Test room join/leave operations."""
        logger.info("Testing room operations...")

        # Create test rooms
        rooms = [f"test_room_{i}" for i in range(3)]

        # Join rooms
        tasks = []
        for i, client in enumerate(self.clients):
            if client.connected:
                room = rooms[i % len(rooms)]
                tasks.append(client.join_room(room))

        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats["messages_sent"] += len(tasks)

        await asyncio.sleep(1)

        # Send chat messages
        tasks = []
        for i, client in enumerate(self.clients):
            if client.connected:
                room = rooms[i % len(rooms)]
                tasks.append(client.send_chat(room, f"Hello from {client.user_id}"))

        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats["messages_sent"] += len(tasks)

        await asyncio.sleep(1)

    async def test_generation_updates(self):
        """Test generation update subscriptions."""
        logger.info("Testing generation updates...")

        # Subscribe to generation channels
        tasks = []
        for client in self.clients[:5]:  # First 5 clients
            if client.connected:
                task_id = f"gen_task_{client.user_id}"
                tasks.append(client.start_generation(task_id, "Test prompt"))

        await asyncio.gather(*tasks, return_exceptions=True)
        self.stats["messages_sent"] += len(tasks)

        await asyncio.sleep(2)

    async def test_concurrent_messaging(self):
        """Test concurrent message sending."""
        logger.info("Testing concurrent messaging...")

        # Send many messages concurrently
        tasks = []
        for _ in range(10):  # 10 rounds
            for client in self.clients:
                if client.connected:
                    tasks.append(client.send_chat("stress_test", f"Message at {time.time()}"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for r in results if not isinstance(r, Exception))
        self.stats["messages_sent"] += successful
        self.stats["errors"] += len(results) - successful

        logger.info(f"Sent {successful}/{len(tasks)} messages successfully")

    async def test_reconnection(self):
        """Test reconnection logic."""
        logger.info("Testing reconnection...")

        # Disconnect some clients
        disconnect_count = min(5, len(self.clients))
        for client in self.clients[:disconnect_count]:
            await client.disconnect()

        await asyncio.sleep(1)

        # Reconnect
        tasks = []
        for client in self.clients[:disconnect_count]:
            tasks.append(client.connect())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        reconnected = sum(1 for r in results if not isinstance(r, Exception))
        logger.info(f"Reconnected {reconnected}/{disconnect_count} clients")

    async def cleanup(self):
        """Disconnect all clients."""
        logger.info("Cleaning up connections...")

        tasks = []
        for client in self.clients:
            if client.connected:
                tasks.append(client.disconnect())

        await asyncio.gather(*tasks, return_exceptions=True)

    def generate_report(self):
        """Generate test report."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Count total messages received
        total_received = sum(len(c.messages_received) for c in self.clients)
        self.stats["messages_received"] = total_received

        print("\n" + "=" * 60)
        print("WEBSOCKET LOAD TEST REPORT")
        print("=" * 60)
        print(f"Test Duration: {duration:.2f} seconds")
        print(f"Clients Created: {len(self.clients)}")
        print(f"Successful Connections: {self.stats['connections_created']}")
        print(f"Messages Sent: {self.stats['messages_sent']}")
        print(f"Messages Received: {self.stats['messages_received']}")
        print(f"Errors: {self.stats['errors']}")
        print(f"Messages/Second: {self.stats['messages_sent']/duration:.2f}")
        print("=" * 60)

        # Message type breakdown
        message_types = {}
        for client in self.clients:
            for msg in client.messages_received:
                msg_type = msg.get("type", "unknown")
                message_types[msg_type] = message_types.get(msg_type, 0) + 1

        print("\nMessage Types Received:")
        for msg_type, count in sorted(message_types.items()):
            print(f"  {msg_type}: {count}")

    async def run_tests(self, client_count: int = 10):
        """Run all tests."""
        self.stats["start_time"] = datetime.utcnow()

        try:
            # Create clients
            await self.create_clients(client_count)

            # Run tests
            await self.test_ping_pong()
            await self.test_room_operations()
            await self.test_generation_updates()
            await self.test_concurrent_messaging()
            await self.test_reconnection()

            # Wait for any remaining messages
            await asyncio.sleep(2)

        finally:
            # Cleanup
            await self.cleanup()
            self.stats["end_time"] = datetime.utcnow()

            # Generate report
            self.generate_report()


async def test_single_client():
    """Test single client functionality."""
    print("\n=== Testing Single Client ===")

    client = WebSocketTestClient()

    try:
        # Connect
        await client.connect()

        # Test basic operations
        await client.ping()
        await asyncio.sleep(0.5)

        await client.join_room("test_room")
        await asyncio.sleep(0.5)

        await client.send_chat("test_room", "Hello, WebSocket!")
        await asyncio.sleep(0.5)

        await client.subscribe_channels(["public:announcements", f"user:{client.user_id}"])
        await asyncio.sleep(0.5)

        # Check messages
        print(f"\nReceived {len(client.messages_received)} messages:")
        for msg in client.messages_received:
            print(f"  - {msg.get('type')}: {msg.get('data', {})}")

    finally:
        await client.disconnect()


async def test_collaboration():
    """Test collaboration features."""
    print("\n=== Testing Collaboration ===")

    # Create multiple clients
    clients = []
    room_id = "collab_room_1"

    try:
        # Create and connect clients
        for i in range(3):
            client = WebSocketTestClient(user_id=f"collab_user_{i}")
            await client.connect()
            await client.join_room(room_id)
            clients.append(client)

        await asyncio.sleep(1)

        # Send collaboration actions
        await clients[0].send_message(
            "collaboration_action",
            {
                "room_id": room_id,
                "action": "edit_track",
                "data": {"position": 10.5, "change": "add_note"},
            },
        )

        await clients[1].send_message(
            "collaboration_action",
            {
                "room_id": room_id,
                "action": "change_tempo",
                "data": {"tempo": 120},
            },
        )

        # Chat
        await clients[0].send_chat(room_id, "Let's make this track awesome!")
        await clients[1].send_chat(room_id, "Sounds good!")
        await clients[2].send_chat(room_id, "I'll add some bass")

        await asyncio.sleep(2)

        # Check received messages
        print("\nCollaboration messages received:")
        for i, client in enumerate(clients):
            collab_msgs = [
                msg
                for msg in client.messages_received
                if msg.get("type") in ["collaboration_update", "chat_message"]
            ]
            print(f"\nClient {i} received {len(collab_msgs)} collaboration messages")

    finally:
        # Cleanup
        for client in clients:
            await client.disconnect()


async def main():
    """Run WebSocket tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test WebSocket implementation")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/api/v1/ws",
        help="WebSocket URL",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "collaboration", "load"],
        default="single",
        help="Test mode",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=10,
        help="Number of clients for load test",
    )

    args = parser.parse_args()

    if args.mode == "single":
        await test_single_client()
    elif args.mode == "collaboration":
        await test_collaboration()
    elif args.mode == "load":
        tester = WebSocketLoadTester(base_url=args.url)
        await tester.run_tests(client_count=args.clients)


if __name__ == "__main__":
    asyncio.run(main())
