import asyncio
import json

from companion import protocol
from companion.server import PetServer


async def connect(server):
    return await asyncio.open_connection("127.0.0.1", server.bound_port)


async def test_pet_receives_messages_as_json_lines_and_companion_gets_pokes():
    got = []
    server = PetServer(port=0, on_message=got.append)
    await server.start()
    reader, writer = await connect(server)
    try:
        writer.write(b'{"t":"hello","version":"1.0"}\n')
        writer.write(b"lixo que nao e json\n")
        writer.write(b'{"t":"poke","kind":"pat"}\n')
        await writer.drain()
        for _ in range(50):
            if len(got) >= 2:
                break
            await asyncio.sleep(0.02)
        assert [m["t"] for m in got] == ["hello", "poke"]

        server.send(protocol.say("u1", "Oi, tudo bem?", "happy"))
        server.send(protocol.mouth(aa=0.5))
        server.send(protocol.end("u1"))
        lines = [json.loads(await asyncio.wait_for(reader.readline(), 2)) for _ in range(3)]
        assert lines[0] == {"t": "say", "id": "u1", "text": "Oi, tudo bem?", "emotion": "happy"}
        assert lines[1]["t"] == "mouth" and lines[2] == {"t": "end", "id": "u1"}
    finally:
        writer.close()
        await server.close()


async def test_send_without_pet_is_a_noop_and_reconnect_works():
    server = PetServer(port=0)
    await server.start()
    try:
        server.send(protocol.end("x"))
        assert not server.connected
        for _ in range(2):
            reader, writer = await connect(server)
            for _ in range(50):
                if server.connected:
                    break
                await asyncio.sleep(0.02)
            server.send(protocol.gesture("wave"))
            assert json.loads(await asyncio.wait_for(reader.readline(), 2)) == {"t": "gesture", "name": "wave"}
            writer.close()
            for _ in range(50):
                if not server.connected:
                    break
                await asyncio.sleep(0.02)
    finally:
        await server.close()
