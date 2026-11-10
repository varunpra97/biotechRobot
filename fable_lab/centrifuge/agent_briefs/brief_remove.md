# Task: remove the tube from the centrifuge

An episode succeeds when the tube is held by the gripper, pulled clear of its bore, lifted about 2 cm above it and
turned upright (within 8 degrees of vertical), and stays so for 10 steps. The verdict comes from the simulator.

Recipes for this task (one model call each):

- `remove_held` : for a start where the gripper is ALREADY closed on the tube (`gripper_commanded` "closed",
  `finger_gap_mm` about 4 to 7). Pulls the tube out along the hand axis, lifts and turns it upright.
- `remove_approach [bore]` : for a start where the gripper is open and the tube sits in a bore. Moves the hand onto
  that bore's tube axis, down to the tube, closes (checks the fingers hold something), pulls out, lifts, turns upright.

Per episode:

1. Look at the start picture and the JSON line (you get both in the reply that ended the previous episode, or from
   `look` at the start of your block).
2. Gripper closed on the tube -> `remove_held`.
3. Gripper open -> find the tube in the rotor (camera B, top right and bottom right, shows it best). Which orange
   ring sits on the tube's open upper end? That is the bore. It is normally the same as `nearest_bore.bore`.
   -> `remove_approach <bore>`.
   If no ring is on the tube's rim (off by more than about two ring widths), say so in your report, and still try
   the ring that is closest.
4. The reply normally contains `episode_over`. Note the outcome and go on with the next episode, whose start picture
   you now have.
5. If the batch stopped early without `episode_over` (for example "REQUIRE FAILED": the fingers closed on nothing),
   look at the picture and recover if `steps_left` is at least about 130: open the gripper and back off with a
   `raw` command, then try `remove_approach` again, perhaps with the other bore. If there are too few steps left,
   call `raw '{"op":"moves","obs":"none","moves":[{"wait":200}]}'` to let the episode run out (it ends as TIMEOUT)
   and continue with the next one.
