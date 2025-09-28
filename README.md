# Follow me, Go there

## Shortcut
```~/.bashrc
alias init_fmgt="robostack_humble && source ~/dev/ROS2/FMGT/install/setup.zsh"
alias fmgt="init_fmgt && ros2 launch fmgt fmgt.launch.py"
alias fmgt_point="init_fmgt && ros2 launch fmgt point_follow.launch.py"
alias fmgt_simulation_point="init_fmgt && ros2 launch fmgt point_simulation.launch.py"
alias fmgt_simulation_path="init_fmgt && ros2 launch fmgt path_simulation.launch.py"
alias fmgt_simulation_remote="init_fmgt && ros2 run fmgt leader_teleop"
```

