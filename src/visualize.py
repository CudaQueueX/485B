import curses
import time


def read_graph_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

    # Parse start and end nodes
    start_node = eval(lines[0].split(":")[1].strip())
    end_node = eval(lines[1].split(":")[1].strip())

    # Parse nodes
    nodes_start = lines.index("nodes:\n") + 1
    path_start = lines.index("path:\n") + 1
    nodes = [
        (int(node[0]), int(node[1]), bool(int(node[2])))  # Convert traversable (1/0) to boolean
        for node in (line.strip("()\n").split(", ") for line in lines[nodes_start:path_start - 1])
    ]

    # Parse path
    path = [eval(line.strip()) for line in lines[path_start:]]

    return start_node, end_node, nodes, path


def draw_controls(stdscr, speed):
    # Display control instructions at the top
    stdscr.addstr(0, 0, "Controls:", curses.A_BOLD)
    stdscr.addstr(1, 0, "[S] Start Animation")
    stdscr.addstr(2, 0, "[R] Reset")
    stdscr.addstr(3, 0, "[Q] Quit")
    stdscr.addstr(4, 0, "[↑] Speed Up  [↓] Slow Down")
    stdscr.addstr(5, 0, f"Current Speed: {speed:.2f} seconds per step")


def draw_graph(stdscr, start_node, end_node, nodes, path=None, animated=False, path_color=5, speed=0.1):
    # Get terminal size
    max_rows, max_cols = stdscr.getmaxyx()

    # Clear previous graph
    stdscr.clear()

    # Draw controls
    draw_controls(stdscr, speed)

    # Determine graph dimensions
    max_x = max(node[0] for node in nodes) + 1
    max_y = max(node[1] for node in nodes) + 1

    # Check if the graph fits in the terminal
    if max_x + 7 > max_rows or max_y * 2 + 5 > max_cols:
        stdscr.addstr(7, 0, "Error: Terminal size too small to display the graph!")
        stdscr.refresh()
        time.sleep(2)
        return

    # Draw nodes
    for x, y, traversable in nodes:
        if (x, y) == start_node:
            stdscr.addstr(x + 7, y * 2, "S", curses.color_pair(1))  # Start node in red
        elif (x, y) == end_node:
            stdscr.addstr(x + 7, y * 2, "T", curses.color_pair(2))  # End node in green
        elif not traversable:
            stdscr.addstr(x + 7, y * 2, "X", curses.color_pair(3))  # Non-traversable in black
        else:
            stdscr.addstr(x + 7, y * 2, ".", curses.color_pair(4))  # Traversable in white

    # Animate the path if requested
    if animated and path:
        for step in path:
            if step != start_node:  # Keep start node visually consistent
                stdscr.addstr(step[0] + 7, step[1] * 2, "*", curses.color_pair(5))  # Path in blue
            # Ensure the start and end nodes remain visible
            stdscr.addstr(start_node[0] + 7, start_node[1] * 2, "S", curses.color_pair(1))
            stdscr.addstr(end_node[0] + 7, end_node[1] * 2, "T", curses.color_pair(2))
            stdscr.refresh()
            time.sleep(speed)
    elif path:
        # Static path
        for step in path:
            if step != start_node:  # Keep start node visually consistent
                stdscr.addstr(step[0] + 7, step[1] * 2, "*", curses.color_pair(path_color))  # Path in a specified color
        # Ensure the start and end nodes remain visible
        stdscr.addstr(start_node[0] + 7, start_node[1] * 2, "S", curses.color_pair(1))
        stdscr.addstr(end_node[0] + 7, end_node[1] * 2, "T", curses.color_pair(2))

    # Refresh screen
    stdscr.refresh()


def main(stdscr):
    # Hide the cursor
    curses.curs_set(0)

    # Initialize curses colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)   # Start node
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK) # End node
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_WHITE) # Non-traversable
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK) # Traversable
    curses.init_pair(5, curses.COLOR_BLUE, curses.COLOR_BLACK)  # Path (animated)
    curses.init_pair(6, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Path (final color)

    # Read graph data
    start_node, end_node, nodes, path = read_graph_data("graph_data.txt")

    # Animation speed
    speed = 0.1  # Default speed (in seconds per step)
    current_path = None  # Keep track of the current path state (None initially)

    # Main loop
    while True:
        # Draw the graph without any path initially
        draw_graph(stdscr, start_node, end_node, nodes, current_path, speed=speed)

        # Wait for user input
        key = stdscr.getch()

        if key in [ord("q"), ord("Q")]:  # Quit
            break
        elif key in [ord("s"), ord("S")]:  # Start animation
            current_path = path
            draw_graph(stdscr, start_node, end_node, nodes, path, animated=True, speed=speed)
            # After animation, update path color
            draw_graph(stdscr, start_node, end_node, nodes, path, animated=False, path_color=6)
        elif key in [ord("r"), ord("R")]:  # Reset
            current_path = None  # Clear the path
            draw_graph(stdscr, start_node, end_node, nodes, current_path, speed=speed)
        elif key == curses.KEY_UP:  # Speed up
            speed = max(0.01, speed - 0.02)  # Reduce delay (min 0.01 seconds)
        elif key == curses.KEY_DOWN:  # Slow down
            speed = min(1.0, speed + 0.02)  # Increase delay (max 1.0 seconds)


if __name__ == "__main__":
    curses.wrapper(main)
