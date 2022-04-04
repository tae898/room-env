from gym.envs.registration import register

register(
    id="RoomEnv-v0",
    entry_point="room_env.envs:RoomEnv",
)
