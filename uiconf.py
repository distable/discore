from PyQt5 import QtCore

hobo_seek_percent = 1 / 15

qkey = QtCore.Qt.Key

key_pause = 'space'
key_cancel = 'espace'
key_reload_script = 'r'
key_reload_script_hard = 's-r'

key_seek_prev = 'left'
key_seek_next = 'right'
key_seek_prev_second = 'up'
key_seek_next_second = 'down'
key_seek_prev_percent = 'pageup'
key_seek_next_percent = 'pagedown'
key_seek_first = 'home', 'h', '0'
key_seek_last = 'n', 'end'

key_fps_down = 'bracketleft'
key_fps_up = 'bracketright'
key_copy_frame = 'c'
key_paste_frame = 'v'
key_delete_frame = 'delete'
key_render_once = 'return'
key_render_forever = 's-return'
key_toggle_hud = 'f'
key_toggle_action_mode = 'w'

key_segment_select_prev = 'less'
key_segment_select_next = 'greater'
key_segment_set_start = 'i'
key_segment_set_end = 'o'
key_seek_prev_segment = 'comma'
key_seek_next_segment = 'period'
key_play_segment = 'p'

key_toggle_ryusig = 'tab', 'escape'

key_dbg_up = 'equal'
key_dbg_down = 'minus'
key_dbg_cycle_increment = 'backslash'
key_open_script_in_editor = 'c-e'
key_open_in_terminal = 'c-s-e'
key_show_session_in_explorer = 's-e'
key_reload_session = 's-r'

key_switch_session_with_cache_reload = 'c-s-r'
key_switch_session = 'c-r'
key_run_action = 'c-a', 'c-s-a'
key_toggle_dev_mode = '2'
key_toggle_trace = '3'
key_toggle_bake_enabled = 'b'

key_snapshot_prev = 's-left'
key_snapshot_next = 's-right'

key_set_last_frame = 's-end'


# ----------------------------------------


key_ryusig_toggle_ryusig = 'tab', 'escape'
key_ryusig_toggle_time = 'f2'
key_ryusig_toggle_pause = 'space'
key_ryusig_toggle_norm = 'tab'
key_ryusig_reframe = '\\'
key_ryusig_dump_timestamps = 'c-s-m'
