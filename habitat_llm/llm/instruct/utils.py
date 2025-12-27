# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

from __future__ import annotations

import typing
from typing import Dict, List, Tuple

import regex as re

if typing.TYPE_CHECKING:
    from habitat_llm.evaluation.evaluation_runner import ActionHistoryElement

import base64
import io

from PIL import Image

from habitat_llm.world_model.world_graph import WorldGraph

PERCEPTION_TOOL_STRINGS = [
    "FindAgentActionTool",
    "FindObjectTool",
    "FindReceptacleTool",
    "FindRoomTool",
]
SINGLE_STEP_PROMPT_HEADER = "Solve the given multi-agent planning problem as best as you can. The task assigned to you will be situated in a house and will generally involve navigating to objects, picking and placing them on different receptacles to achieve rearrangement. Below is the detailed description of the actions you can use for solving the task."
STOP_WORD = "<end_act>"


def get_world_descr(
    world_graph,
    agent_uid=0,
    include_room_name=False,
    add_state_info=False,
    centralized=False,
):
    """
    Builds a string description of the environment from the world graph for single step planners.

    :param world_graph: The world graph representing the environment.
    :return: A string description of the environment, including rooms and their furniture, objects held by the agent, and locations of objects in the house.
    """
    ## house description -- rooms and their furniture list
    furn_room = world_graph.group_furniture_by_room()
    house_info = ""
    for k, v in furn_room.items():
        furn_names = [furn.name for furn in v]
        all_furn = ", ".join(furn_names)
        house_info += k + ": " + all_furn + "\n"

    all_furniture = world_graph.get_all_furnitures()
    furn_with_faucets = [
        fur for fur in all_furniture if "faucet" in fur.properties.get("components", [])
    ]
    faucet_info = "The following furnitures have a faucet: " + ", ".join(
        [fur.name for fur in furn_with_faucets]
    )

    objs_info = get_objects_descr(
        world_graph, agent_uid, include_room_name, add_state_info, centralized
    )
    return f"Furniture:\n{house_info}\n{faucet_info}\nObjects:\n{objs_info}"


def state_dict_to_string(state_dict):
    """
    Transforms a state dictionary into a human-readable string.

    :param state_dict: A dictionary of states, e.g., {'is_clean': False, 'is_powered_on': True}
    :return: A string describing the states, e.g., "is not clean and is powered on"
    """
    state_strings = []

    for state, value in state_dict.items():
        if state.startswith("is_"):
            state = state[3:]  # Remove 'is_' prefix
        state = state.replace("_", " ")
        state_strings.append(f"{state}: {value}")
    return ", ".join(state_strings)


def get_objects_descr(
    world_graph,
    agent_uid=0,
    include_room_name=False,
    add_state_info=False,
    centralized=False,
):
    """
    Builds a string description of objects in the environment.

    :param world_graph: The world graph representing the environment.
    :return: A string description of objects, including their locations.
    """

    # lazy import to avoid importing habitat module if not necessary
    from habitat_llm.world_model.entity import Room

    all_objs = world_graph.get_all_objects()
    if not all_objs:
        return "No objects found yet"
    else:
        obj_strings = []
        for obj in all_objs:
            obj_info = ""
            rooms_path = world_graph.find_path(root_node=obj, end_node_types=[Room])
            if rooms_path is None:
                room_name = "an unknown room"
            else:
                rooms = [x for x in rooms_path if isinstance(x, Room)]
                if len(rooms) == 0:
                    room_name = "an unknown room"
                else:
                    if len(rooms) > 1:
                        raise ValueError(
                            f"Multiple rooms detected for object {obj.name}"
                        )
                    room_name = rooms[0].name
            if not centralized and (
                world_graph.is_object_with_robot(obj)
                and int(agent_uid) == 0
                or (world_graph.is_object_with_human(obj) and int(agent_uid) == 1)
            ):
                obj_info += obj.name + ": held by the agent"
            elif not centralized and (
                world_graph.is_object_with_human(obj)
                and int(agent_uid) == 0
                or (world_graph.is_object_with_robot(obj) and int(agent_uid) == 1)
            ):
                obj_info += obj.name + ": held by the other agent"
            elif centralized and world_graph.is_object_with_robot(obj):
                obj_info += obj.name + ": held by Agent 0 (Robot)"
            elif centralized and world_graph.is_object_with_human(obj):
                obj_info += obj.name + ": held by Agent 1 (Human)"
            else:
                furn_node = world_graph.find_furniture_for_object(obj)
                furn_name = "unknown" if furn_node is None else furn_node.name
                if include_room_name:
                    obj_info += obj.name + ": " + furn_name + " in " + room_name
                else:
                    obj_info += obj.name + ": " + furn_name
            if (add_state_info) and ("states" in obj.properties):
                state_string = state_dict_to_string(obj.properties["states"])
                if len(state_string) > 0:
                    obj_info += ". States: " + state_string
            obj_strings.append(obj_info)
        return "\n".join(obj_strings)


def get_rearranged_objects_descr(
    obj_descr_t_1,
    obj_descr_t,
):
    """
    Builds a string description of objects that were updated by latest agent actions execution

    :param object description string at t-1 and at t
    :return: A string description of objects that are updated by agent actions, including their updated locations.
    """

    objs_list_t_1 = obj_descr_t_1.split("\n")
    objs_list_t = obj_descr_t.split("\n")
    updated_objs = []

    objs_t_1 = any(":" in s for s in objs_list_t_1)
    objs_t = any(":" in s for s in objs_list_t)

    if not objs_t_1 and objs_t:
        return "\n".join(objs_list_t)

    if len(objs_list_t) == len(objs_list_t_1):
        for obj_id, obj in enumerate(objs_list_t):
            if obj != objs_list_t_1[obj_id]:
                updated_objs.append(obj)
    else:
        updated_objs = [obj for obj in objs_list_t if obj not in objs_list_t_1]

    return "\n".join(updated_objs)


def build_single_step_prompt(
    task: str,
    world_graph: WorldGraph,
    agent_uid: str,
    action_history: Dict[str, List[ActionHistoryElement]],
    action_representation=True,
    tools_to_skip=None,
):
    """
    Constructs the prompt for a single step planner.

    This function gathers all actions of the specified agent and all actions of other agents from the environment interface's action history or state history based on the action_representation parameter.

    The function assumes that the environment interface's action history and state history contain actions and states of exactly two agents with UIDs '0' and '1'.

    :param task: The instruction assigned to the agent.
    :param world_graph: The current environment world graph.
    :param agent_uid: The UID of the agent for which to construct the prompt.
    :param action_representation: Boolean flag to determine whether to use action history or state history.
    :return: The constructed prompt.
    """
    if tools_to_skip is None:
        tools_to_skip = []
    all_actions: List[ActionHistoryElement] = []
    if action_representation:
        all_actions = sum(action_history.values(), [])
    else:
        raise NotImplementedError("State history not implemented")
        # for agent, actions in action_history.items():
        #     if int(agent) == int(agent_uid):
        #         all_actions.extend(actions)
        # # Extract other agent's actions from agent_state_history
        # for agent, states in agent_state_history.items():
        #     if int(agent) != int(agent_uid):
        #         all_actions.extend(states)

    prev_action_string = ""
    all_actions.sort(key=lambda x: x.timestamp)
    # remove perception tools from the list of actions
    all_actions = [
        action
        for action in all_actions
        if not hasattr(action, "action") or action.action[0] not in tools_to_skip
    ]
    strings = []
    for action in all_actions:
        if int(action.agent_uid) == int(agent_uid):
            strings.append(f"Agent_Action: {action.to_string()}")
            strings.append(f"Action Result: {action.response}")
        elif (
            "navigate" not in action.to_string().lower()
            and "find" not in action.to_string().lower()
            and "explore" not in action.to_string().lower()
        ):
            # We dont want to add Explore actions
            strings.append(f"Other_Agent_Action: {action.to_string()}")

    prev_action_string = (
        "\n".join(strings) if len(strings) > 0 else "No previous actions taken"
    )

    world_graph_string = get_world_descr(world_graph)
    task_text = f"Task: {task}"
    graph_text = f"Current Environment:\n{world_graph_string}"
    previous_actions = "Previous actions:\n" + prev_action_string
    curr_action_str = "Next Agent_Action:<|reserved_special_token_0|>"
    new_text = "\n\n".join(
        [SINGLE_STEP_PROMPT_HEADER]
        + [task_text, graph_text, previous_actions, curr_action_str]
    )
    return new_text


def zero_shot_prompt_action_parser(text):
    """
    This method parses the llm response to extract
    Action (tool name) and Action Input which can be
    used to execute a specific action via Tool
    """
    regex = r"Action: (.*?)[\n]*Action Input: (.*)"
    match = re.search(regex, text, re.DOTALL)
    if not match:
        print(text)
        raise ValueError(f"Could not parse LLM output: `{text}`")
    action = match.group(1).strip()
    action_input = match.group(2)
    # Remove quotes and whitespace and \n
    return action, action_input.strip(" ").strip('"').strip("\n")


def action_in_brackets_parser(text):
    """
    This method parses the llm response to extract
    Action (tool name) and Action Input from a string with bracketed syntax.
    e.g. "NavigateSkill[sofa_0]" -> "NavigateSkill", "sofa_0"
    """
    action_prefix = "Action: "

    if not text.split("\n")[-1].startswith(action_prefix):
        return None

    action_block = text.split("\n")[-1]
    action_str = action_block[len(action_prefix) :]

    # Parse out the action and the directive.
    re_matches = re.search(r"(.*?)\[(.*?)\]", action_str)
    if re_matches is None:
        raise ValueError(f"Could not parse action directive: {action_str}")

    return re_matches.group(1), re_matches.group(2)


def zero_shot_prompt_agent_action_parser(text):
    """
    This method parses the llm response to extract
    Action (tool name) and Action Input and Agent uid from a string.
    """
    regex = r"Action: (.*?)[\n]*Action Input: (.*)[\n]*Agent: (.*)"
    match = re.search(regex, text, re.DOTALL)
    if not match:
        print(text)
        raise ValueError(f"Could not parse LLM output: `{text}`")
    action = match.group(1).strip()
    action_input = match.group(2).strip()
    agent = match.group(3)

    # Remove quotes and whitespace and \n
    return agent.strip(" ").strip('"').strip("\n"), action, action_input


def has_valid_square_brackets(input_string):
    return "[" in input_string and "]" in input_string


def split_string(input_string, delimiter=","):
    # Early return if string does not have delimiter
    if delimiter not in input_string:
        return input_string

    # Remove spaces
    input_string_filtered = input_string.replace(" ", "")

    # Split at delimiter
    substrings = input_string_filtered.split(delimiter)

    return substrings


def most_matching_string(input_str, candidate_strings):
    """
    Method to get most matching string
    """

    # Remove non-alphabetical characters from the input string and candidate strings
    input_str_filtered = "".join(filter(str.isalnum, input_str))
    candidate_strings_stripped = [
        "".join(filter(str.isalnum, cand)) for cand in candidate_strings
    ]

    # Initialize variables to keep track of the most matching string and the number of matching characters
    max_matching_string = ""
    max_matching_chars = 0

    # Loop through each candidate string and count the number of matching characters
    for i, cand in enumerate(candidate_strings_stripped):
        matching_chars = len(
            list(filter(lambda x: x[0] == x[1], zip(input_str_filtered, cand)))
        )
        if matching_chars > max_matching_chars:
            max_matching_string = candidate_strings[i]
            max_matching_chars = matching_chars

    return max_matching_string


def fetch_from_valid_search_space(action_name, action_input, agent_id, params=None):
    # Return inputs as they are if params is none
    if params == None:
        return action_name, action_input

    # Get the best matching action name
    tool_list = list(params["tool_list"])
    action_name_corrected = most_matching_string(action_name, tool_list)

    # Get the best matching action input
    motor_actions = ["navigate", "open", "close", "rearrange", "pick", "place"]
    if any(motor_action in action_name.lower() for motor_action in motor_actions):
        substrings = split_string(action_input, ",")
        valid_node_names = params["world_graph"][agent_id].get_all_node_names()
        action_input_corrected = ", ".join(
            most_matching_string(substr, valid_node_names) for substr in substrings
        )
    else:
        action_input_corrected = action_input

    return action_name_corrected, action_input_corrected


def remove_non_alpha_left(input_string):
    """
    This method strips non alphabetical characters from the left part of the string until first alphabetical character is found.
    Useful to handle cases such as, '- Agent', '** Agent_' ' Agent_' etc. which are
    the result of LLM not following the correct syntax.
    """
    for i, char in enumerate(input_string):
        if char.isalpha():
            return input_string[i:]
    return ""


def zero_shot_action_parser(agents, input_string, params=None):
    # get the skill call before the Assigned!
    action_line = input_string.strip().split("\n")[-1]

    # this parser is for single agent only
    assert len(agents) == 1
    agent_id = agents[0].uid
    # reuse the existing parser
    return actions_parser(agents, f"Agent_{agent_id}_Action: {action_line}", params)


def zero_shot_action_parser_qwen(agents, input_string, params=None):
    """
    Qwen-specific action parser that handles the case where the model generates
    responses with 'Assigned!' or 'Assigned' at the end, but the actual action
    is in the line before it.
    Also filters out ChatML tags that Qwen sometimes outputs.
    """
    # Remove ChatML tags that Qwen might output
    input_string = input_string.replace("<|im_start|>", "")
    input_string = input_string.replace("<|im_end|>", "")
    input_string = input_string.replace("<|im_start|>assistant", "")
    input_string = input_string.replace("<|im_start|>user", "")

    lines = input_string.strip().split("\n")

    # Filter out lines that are just ChatML tags or empty
    filtered_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip lines that are only ChatML tags or whitespace
        if (
            line_stripped
            and not line_stripped.startswith("<|")
            and not line_stripped.endswith("|>")
        ):
            filtered_lines.append(line)

    if not filtered_lines:
        filtered_lines = lines

    # Find the action line - look for the line that contains square brackets
    action_line = None
    for line in reversed(filtered_lines):
        line_stripped = line.strip()
        # Remove any remaining ChatML tags from the line
        line_stripped = (
            line_stripped.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        )
        if "[" in line_stripped and "]" in line_stripped:
            action_line = line_stripped
            break

    # If no line with brackets found, use the last non-empty line
    if action_line is None:
        for line in reversed(filtered_lines):
            line_stripped = (
                line.strip()
                .replace("<|im_start|>", "")
                .replace("<|im_end|>", "")
                .strip()
            )
            if line_stripped:
                action_line = line_stripped
                break

    # If still no action line, use the last line
    if action_line is None and filtered_lines:
        action_line = (
            filtered_lines[-1]
            .strip()
            .replace("<|im_start|>", "")
            .replace("<|im_end|>", "")
            .strip()
        )

    # this parser is for single agent only
    assert len(agents) == 1
    agent_id = agents[0].uid
    # reuse the existing parser
    return actions_parser(agents, f"Agent_{agent_id}_Action: {action_line}", params)


def actions_parser(
    agents, input_string, params=None
) -> Dict[int, Tuple[str, str, str]]:
    """
    Actions parser used by planners to convert LLM generation
    into a structured representation.
    """

    # Container to store parser output
    actions_dict: Dict[int, Tuple[str, str, str]] = {}

    # Split input string
    lines = input_string.strip().split("\n")

    for line in lines:
        line = line.strip()
        line = remove_non_alpha_left(line)
        if line.startswith("Agent") and ("_Action" in line):
            # Extract agent info and actions info
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue

            agent_id, action_info = parts[0].strip(), parts[1].strip()

            # Extracting the numerical part of the agent ID
            if "_" in agent_id:
                agent_id = int(agent_id.split("_")[1])
            else:
                agent_id_list = [int(i) for i in parts[0].split() if i.isdigit()]
                if len(agent_id_list) < 1:
                    continue
                agent_id = agent_id_list[0]

            # Make sure that agent uid is valid
            true_agent_ids = [agent.uid for agent in agents]
            if agent_id not in true_agent_ids:
                for true_agent_id in true_agent_ids:
                    actions_dict[true_agent_id] = (
                        None,
                        None,
                        f"Invalid Agent ID in Action directive. Only valid Agent IDs are {true_agent_ids}!",
                    )
                continue

            # Make syntax exception for Wait command
            if "Wait" in action_info:
                action_info = "Wait[]"

            # Add error message to indicate if the line does not have complete square brackets
            if not has_valid_square_brackets(action_info):
                actions_dict[agent_id] = (
                    None,
                    None,
                    'SyntaxError in Action directive. Opening "[" or closing "]" square bracket is missing!',
                )
                continue

            # Add error message for invalid actions
            if agent_id == 0:
                if "Fill" in action_info:
                    actions_dict[agent_id] = (
                        None,
                        None,
                        "Your agent cannot fill objects. You should let your partner agent do this part of the task and move on to other parts of the task. If you are holding a corresponding object for this action, please place it on a receptacle with faucet.",
                    )
                elif "Clean" in action_info:
                    actions_dict[agent_id] = (
                        None,
                        None,
                        "Your agent cannot clean or wash objects. You should let your partner agent do this part of the task and move on to other parts of the task. If you are holding a corresponding object for this action, please place it on the floor and proceed to other parts of the task.",
                    )
                elif "PoweredOn" in action_info:
                    actions_dict[agent_id] = (
                        None,
                        None,
                        "Your agent cannot turn on objects. You should let your partner agent do this part of the task and move on to other parts of the task",
                    )
                elif "PoweredOff" in action_info:
                    actions_dict[agent_id] = (
                        None,
                        None,
                        "Your agent cannot turn off objects. You should let your partner agent do this part of the task and move on to other parts of the task",
                    )
                else:
                    if params and not any(
                        tool in action_info for tool in params["tool_list"]
                    ):
                        actions_dict[agent_id] = (
                            None,
                            None,
                            "This tool/action is invalid for your agent. So no actions will be assigned to the agent. Please re-think what your agent should do for the task and assign a valid action to the agent.",
                        )
            else:
                if params and not any(
                    tool in action_info for tool in params["tool_list"]
                ):
                    actions_dict[agent_id] = (
                        None,
                        None,
                        "This tool/action is invalid for your agent. So no actions will be assigned to the agent. Please re-think what your agent should do for the task and assign a valid action to the agent.",
                    )

            # Split the action info into action name and action arguments (inputs)
            action_name, action_input = action_info.split("[")
            action_input = action_input.rstrip("]")

            # Set action_input to None if its empty
            # Useful in handling cases like Wait[], FindAgentAction[]
            if action_input == "":
                action_input = None

            actions_dict[agent_id] = (action_name, action_input, None)

    return actions_dict


def finetuned_actions_parser(
    agent_id: int, agents: dict, input_string: str, params=None
):
    """
    Parses an input string to extract an action and its parameters.

    Input strings will be in the format "action[parameters]".
    :param input_string: The input string to parse.
    :param params: Unused to match the signature of other action parsers
    :param agent_id: The unique identifier of the agent performing the action.
    """
    assert (
        agent_id is not None
    ), "Agent ID must be provided for finetuned actions parser."
    pattern = r"(\w+)\[(.*?)\]"
    matches = re.match(pattern, input_string)
    if matches is None:
        return {}
    else:
        return {agent_id: (matches[1], matches[2], None)}


def image_data_to_pil(data: str):
    header = "data:image/png;base64,"
    return Image.open(
        io.BytesIO(base64.decodebytes(bytes(data.split(header)[1], "utf-8")))
    )


def pil_image_to_data_url(image: Image):
    """
    Convert array to numpy64
    """
    # Guess the MIME type of the image based on the file extension
    # Note this can be obtained using pil_image = Image.fromarray(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="png")
    img_byte = buffered.getvalue()
    base64_encoded_data = base64.b64encode(img_byte).decode("utf-8")

    mime_type = "image/png"

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def enhanced_skill_action_parser(agents, input_string, params=None):
    """
    Enhanced action parser for skill-pattern based prompts.
    This is a wrapper around zero_shot_action_parser that handles
    skill pattern annotations in the output and extracts the correct action.
    """
    # Remove skill pattern annotations if present
    lines = input_string.strip().split("\n")
    filtered_lines = []

    for line in lines:
        # Skip lines that are skill pattern commentary or explanatory text
        line_stripped = line.strip()
        if (
            line_stripped.startswith("Skill Pattern:")
            or line_stripped.startswith("Decision Strategy:")
            or line_stripped.startswith("allokay")
            or line_stripped.startswith("assistantin")
            or line_stripped.startswith("Here is")
            or line_stripped.startswith("I'll proceed")
            or line_stripped.startswith("I should have")
        ):
            continue
        filtered_lines.append(line)

    filtered_input = "\n".join(filtered_lines)

    # Try to find the action line with square brackets
    # Look for pattern: Thought: ... followed by Action[param] followed by Assigned!
    action_line = None
    for _i, line in enumerate(filtered_lines):
        line_stripped = line.strip()
        # Check if this line contains an action with square brackets
        if "[" in line_stripped and "]" in line_stripped:
            # Check if it's a valid action format (Action[param])
            # Extract the last token which should be Action[param]
            tokens = line_stripped.split()
            if tokens:
                last_token = tokens[-1]
                if re.match(r"^\w+\[.*?\]", last_token):
                    action_line = (
                        last_token  # Take the last word which should be Action[param]
                    )
                    break

    # If found a valid action line, use it directly
    if action_line:
        # this parser is for single agent only
        assert len(agents) == 1
        agent_id = agents[0].uid
        return actions_parser(agents, f"Agent_{agent_id}_Action: {action_line}", params)

    # Fallback to standard parser
    return zero_shot_action_parser(agents, filtered_input, params)


def enhanced_skill_rag_retriever(
    query,
    top_k=15,
    similarity_threshold=0.75,
    skill_pattern_boost=1.2,
    quality_threshold=0.7,
    coordination_aware=True,
    efficiency_weighted=True,
):
    """
    Enhanced RAG retriever that considers skill patterns, quality metrics,
    and coordination requirements.

    This is a placeholder that returns the same interface as the standard retriever.
    The actual filtering logic should be implemented in the RAG module.
    """
    # For now, just return the parameters as a config dict
    # The actual retrieval will be handled by the RAG system
    return {
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "skill_pattern_boost": skill_pattern_boost,
        "quality_threshold": quality_threshold,
        "coordination_aware": coordination_aware,
        "efficiency_weighted": efficiency_weighted,
    }


def extract_object_states_from_world_graph(
    world_graph,
    agent_uid=0,
    include_room_name=False,
    centralized=False,
) -> Dict[str, Dict]:
    """
    直接从 world_graph 提取对象的状态信息（位置、状态属性等），返回结构化数据。

    :param world_graph: The world graph representing the environment
    :param agent_uid: Agent UID
    :param include_room_name: Whether to include room name in location
    :param centralized: Whether this is centralized planning
    :return: Dictionary mapping object names to their state info:
             {obj_name: {"location": str, "states": dict}}
    """
    from habitat_llm.world_model.entity import Room

    objects_state = {}
    all_objs = world_graph.get_all_objects()

    if not all_objs:
        return objects_state

    for obj in all_objs:
        obj_info = {"location": None, "states": {}}

        # Get room name
        rooms_path = world_graph.find_path(root_node=obj, end_node_types=[Room])
        room_name = None
        if rooms_path is not None:
            rooms = [x for x in rooms_path if isinstance(x, Room)]
            if len(rooms) > 0:
                if len(rooms) > 1:
                    raise ValueError(f"Multiple rooms detected for object {obj.name}")
                room_name = rooms[0].name

        # Determine location
        location = None
        if not centralized and (
            world_graph.is_object_with_robot(obj)
            and int(agent_uid) == 0
            or (world_graph.is_object_with_human(obj) and int(agent_uid) == 1)
        ):
            location = "held by the agent"
        elif not centralized and (
            world_graph.is_object_with_human(obj)
            and int(agent_uid) == 0
            or (world_graph.is_object_with_robot(obj) and int(agent_uid) == 1)
        ):
            location = "held by the other agent"
        elif centralized and world_graph.is_object_with_robot(obj):
            location = "held by Agent 0 (Robot)"
        elif centralized and world_graph.is_object_with_human(obj):
            location = "held by Agent 1 (Human)"
        else:
            furn_node = world_graph.find_furniture_for_object(obj)
            furn_name = "unknown" if furn_node is None else furn_node.name
            if include_room_name and room_name:
                location = f"{furn_name} in {room_name}"
            else:
                location = furn_name

        obj_info["location"] = location

        # Extract states from properties
        if "states" in obj.properties:
            obj_info["states"] = dict(obj.properties["states"])

        objects_state[obj.name] = obj_info

    return objects_state


def compute_state_differences_from_world_graphs(
    previous_objects_state: Dict[str, Dict],
    current_objects_state: Dict[str, Dict],
) -> str:
    """
    直接比较两个 world_graph 的状态字典，找出变化。
    这是更可靠的方法，避免了文本解析的问题。

    :param previous_objects_state: 之前的状态字典 {obj_name: {"location": str, "states": dict}}
    :param current_objects_state: 当前的状态字典 {obj_name: {"location": str, "states": dict}}
    :return: Formatted string describing state changes
    """
    if not previous_objects_state:
        return "This is your first observation. You have just started the task - NO objects have been moved or placed yet. You MUST explore the environment first to find the objects mentioned in the task, then navigate to them, pick them up, and place them in the target locations. Do NOT use Done[] - the task has not even begun. You need to complete all the required object movements before the task is done."

    changes = []

    # Check existing objects for changes
    for obj_name, curr_info in current_objects_state.items():
        if obj_name in previous_objects_state:
            prev_info = previous_objects_state[obj_name]

            # Check location changes
            if prev_info["location"] != curr_info["location"]:
                changes.append(
                    f"  • {obj_name}: MOVED from '{prev_info['location']}' to '{curr_info['location']}'"
                )

            # Check state property changes
            prev_states = prev_info.get("states", {})
            curr_states = curr_info.get("states", {})
            state_changes = []
            all_states = set(prev_states.keys()) | set(curr_states.keys())
            for state_key in all_states:
                prev_value = prev_states.get(state_key)
                curr_value = curr_states.get(state_key)
                if prev_value != curr_value:
                    prev_str = str(prev_value) if prev_value is not None else "None"
                    curr_str = str(curr_value) if curr_value is not None else "None"
                    state_changes.append(f"{state_key}: {prev_str} → {curr_str}")

            if state_changes:
                changes.append(
                    f"  • {obj_name}: STATE CHANGED - {', '.join(state_changes)}"
                )
        else:
            # Newly discovered object
            state_info = ""
            if curr_info.get("states"):
                state_items = [f"{k}: {v}" for k, v in curr_info["states"].items()]
                state_info = f" (States: {', '.join(state_items)})"
            changes.append(
                f"  • {obj_name}: NEWLY DISCOVERED at '{curr_info['location']}'{state_info}"
            )

    # Objects that disappeared
    for obj_name, prev_info in previous_objects_state.items():
        if obj_name not in current_objects_state:
            changes.append(
                f"  • {obj_name}: REMOVED from view (was at '{prev_info['location']}')"
            )

    if not changes:
        return "No state changes detected since last observation. Continue working on the task - do not use Done[] unless all objectives are completed."

    state_diff = "State Changes Detected:\n" + "\n".join(changes)

    # Add interpretation hints
    state_diff += "\n\nInterpretation Hints:"
    if any(
        "held by other agent" in change or "held by the other agent" in change
        for change in changes
    ):
        state_diff += "\n  → Other agent is currently holding an object"
    if any("MOVED" in change for change in changes):
        state_diff += "\n  → Other agent likely performed pick and/or place actions"
    if any("NEWLY DISCOVERED" in change for change in changes):
        state_diff += (
            "\n  → Other agent likely explored a new area or opened containers"
        )
    if any("STATE CHANGED" in change for change in changes):
        state_diff += "\n  → Object properties were modified (e.g., cleaned, powered on/off, filled)"
        if any(
            "clean" in change.lower() for change in changes if "STATE CHANGED" in change
        ):
            state_diff += "\n    - An object was cleaned (clean: False → True)"
        if any(
            "powered" in change.lower()
            for change in changes
            if "STATE CHANGED" in change
        ):
            state_diff += "\n    - An object was powered on/off"

    return state_diff


def parse_object_description_line(line: str) -> tuple:
    """
    Parse an object description line to extract object name, location, and states.

    Example input: "apple_0: counter_0. States: clean: False, powered_on: True"
    Returns: ("apple_0", "counter_0", {"clean": False, "powered_on": True})

    :param line: Object description line
    :return: Tuple of (object_name, location, states_dict)
    """
    obj_name = None
    location = None
    states = {}

    if ":" not in line:
        return None, None, {}

    # Split by "States:" to separate location from states
    if ". States:" in line:
        parts = line.split(". States:", 1)
        location_part = parts[0].strip()
        states_part = parts[1].strip()

        # Parse states: "clean: False, powered_on: True"
        if states_part:
            state_items = states_part.split(",")
            for item in state_items:
                item = item.strip()
                if ":" in item:
                    state_parts = item.split(":", 1)
                    if len(state_parts) == 2:
                        state_key = state_parts[0].strip()
                        state_value_str = state_parts[1].strip()
                        # Convert string to boolean/int if possible
                        if state_value_str.lower() == "true":
                            state_value = True
                        elif state_value_str.lower() == "false":
                            state_value = False
                        elif state_value_str.isdigit():
                            state_value = int(state_value_str)
                        else:
                            state_value = state_value_str
                        states[state_key] = state_value
    else:
        location_part = line

    # Parse object name and location: "apple_0: counter_0"
    if location_part and ":" in location_part:
        name_parts = location_part.split(":", 1)
        if len(name_parts) == 2:
            obj_name = name_parts[0].strip()
            location = name_parts[1].strip()

    return obj_name, location, states


def compute_state_differences(previous_obj_descr: str, current_obj_descr: str) -> str:
    """
    Compares previous and current object descriptions to identify state changes.
    Returns a formatted string highlighting the differences, including location changes
    and state property changes (e.g., is_clean, is_powered_on).

    :param previous_obj_descr: Previous object description string
    :param current_obj_descr: Current object description string
    :return: Formatted string describing state changes
    """
    if not previous_obj_descr or previous_obj_descr == "No objects found yet":
        return "This is your first observation. You have just started the task - NO objects have been moved or placed yet. You MUST explore the environment first to find the objects mentioned in the task, then navigate to them, pick them up, and place them in the target locations. Do NOT use Done[] - the task has not even begun. You need to complete all the required object movements before the task is done."

    # Parse object descriptions with states
    prev_objects = {}  # {obj_name: (location, states_dict)}
    curr_objects = {}  # {obj_name: (location, states_dict)}

    for line in previous_obj_descr.split("\n"):
        if line.strip() and ":" in line:
            obj_name, location, states = parse_object_description_line(line)
            if obj_name:
                prev_objects[obj_name] = (location, states)

    for line in current_obj_descr.split("\n"):
        if line.strip() and ":" in line:
            obj_name, location, states = parse_object_description_line(line)
            if obj_name:
                curr_objects[obj_name] = (location, states)

    # Identify changes
    changes = []

    # Objects that moved or changed state
    for obj_name, (curr_location, curr_states) in curr_objects.items():
        if obj_name in prev_objects:
            prev_location, prev_states = prev_objects[obj_name]

            # Check location changes
            if prev_location != curr_location:
                changes.append(
                    f"  • {obj_name}: MOVED from '{prev_location}' to '{curr_location}'"
                )

            # Check state changes
            state_changes = []
            all_states = set(prev_states.keys()) | set(curr_states.keys())
            for state_key in all_states:
                prev_value = prev_states.get(state_key)
                curr_value = curr_states.get(state_key)
                if prev_value != curr_value:
                    prev_str = str(prev_value) if prev_value is not None else "None"
                    curr_str = str(curr_value) if curr_value is not None else "None"
                    state_changes.append(f"{state_key}: {prev_str} → {curr_str}")

            if state_changes:
                changes.append(
                    f"  • {obj_name}: STATE CHANGED - {', '.join(state_changes)}"
                )
        else:
            state_info = ""
            if curr_states:
                state_items = [f"{k}: {v}" for k, v in curr_states.items()]
                state_info = f" (States: {', '.join(state_items)})"
            changes.append(
                f"  • {obj_name}: NEWLY DISCOVERED at '{curr_location}'{state_info}"
            )

    # Objects that disappeared (unlikely but possible)
    for obj_name, (prev_location, prev_states) in prev_objects.items():
        if obj_name not in curr_objects:
            changes.append(
                f"  • {obj_name}: REMOVED from view (was at '{prev_location}')"
            )

    if not changes:
        return "No state changes detected since last observation. Continue working on the task - do not use Done[] unless all objectives are completed."

    state_diff = "State Changes Detected:\n" + "\n".join(changes)

    # Add interpretation hints
    state_diff += "\n\nInterpretation Hints:"
    if any(
        "held by other agent" in change or "held by the other agent" in change
        for change in changes
    ):
        state_diff += "\n  → Other agent is currently holding an object"
    if any("MOVED" in change for change in changes):
        state_diff += "\n  → Other agent likely performed pick and/or place actions"
    if any("NEWLY DISCOVERED" in change for change in changes):
        state_diff += (
            "\n  → Other agent likely explored a new area or opened containers"
        )
    if any("STATE CHANGED" in change for change in changes):
        state_diff += "\n  → Object properties were modified (e.g., cleaned, powered on/off, filled)"
        if any(
            "clean" in change.lower() for change in changes if "STATE CHANGED" in change
        ):
            state_diff += "\n    - An object was cleaned (clean: False → True)"
        if any(
            "powered" in change.lower()
            for change in changes
            if "STATE CHANGED" in change
        ):
            state_diff += "\n    - An object was powered on/off"

    return state_diff
