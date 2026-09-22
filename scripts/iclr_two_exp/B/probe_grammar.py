"""Probe: does the vLLM endpoint accept the planner's transformers-CFG grammar as a structured-output grammar?
Builds a grammar with the same text form as LLMPlanner.build_response_grammar / build_tool_grammar and sends one
greedy request. Usage: probe_grammar.py <base_url>"""
import json, sys, requests

url = sys.argv[1].rstrip("/")
FREE = 'free_text ::= [ "\'.:,!a-zA-Z_0-9]*'
g = "\n".join([
    'root ::= free_text "\\n" (action | termination)',
    'action ::= action_0 "\\nAssigned!"',
    'action_0 ::= "Agent_0_Action: " tool_call',
    'termination ::= "Final Thought: Exit!"',
    "tool_call ::= Navigate | Pick | Place | Explore | Wait | Done",
    'Navigate ::= "Navigate[" nav_target "]"',
    'Pick ::= "Pick[" object "]"',
    'Place ::= "Place[" object "," WS spatial_relation "," WS furniture "," WS spatial_constraint "," WS obj_or_furniture "]"',
    'Explore ::= "Explore[" room "]"',
    'Wait ::= "Wait[]"',
    'Done ::= "Done[]"',
    "nav_target ::= (furniture | room | object)",
    'object ::= "cup_0" | "jug_1"',
    "obj_or_furniture ::= (furniture | object)",
    'furniture ::= "table_12" | "counter_29"',
    'room ::= "kitchen_1" | "living_room_1"',
    'spatial_constraint ::= "next_to"',
    'spatial_relation ::= "on" | "within"',
    FREE,
    "WS ::= [ ]*",
])
prompt = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTask: move the cup to the table in the kitchen. "
          "Respond with a thought line, then the action.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThought:")
for key in ("structured_outputs", "guided_grammar"):
    body = {"model": "llama31-8b", "prompt": prompt, "max_tokens": 200, "temperature": 0, "stop": ["Assigned!"]}
    body[key] = {"grammar": g} if key == "structured_outputs" else g
    r = requests.post(url + "/completions", json=body, timeout=300)
    print(key, r.status_code, json.dumps(r.json())[:600])
body = {"model": "llama31-8b", "prompt": prompt, "max_tokens": 200, "temperature": 0, "stop": ["Assigned!"]}
r = requests.post(url + "/completions", json=body, timeout=300)
print("free", r.status_code, json.dumps(r.json()["choices"][0]["text"])[:400])
