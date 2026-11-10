# Task: insert the held tube into a bore of the centrifuge rotor

Every episode starts with the tube in the closed gripper. A green circle in the top-left wide picture of the start
picture marks the target bore. The episode succeeds when the tube is seated in that bore (deep, within 1.5 mm and
5 degrees of the bore axis), the gripper is open and the hand has backed off 2 cm. The verdict comes from the
simulator. The bore is only about 0.7 mm wider than the tube all round, so the tube must be lined up before the push.

The catch: the tube is often NOT straight in the fingers (it can be several degrees and several millimetres off the
hand's axis). The hand pose is known exactly; where the tube really is you only know from the pictures.

Recipes for this task (one model call each):

- `insert_align [bore]` : puts the HAND on the bore axis with the tube tip (if the tube were straight in the hand)
  4 mm above the bore opening. Returns two 4x crops side by side: left = camera A (side), right = camera B (from
  above). Two magenta rings in each crop: the one nearer the tube is **m0 = where the tube's pointed tip should be**;
  the other, m1, is the bore opening. Ring m0 is about 3 mm across (about 32 picture pixels).
- `insert_fix [bore] dxA dyA dxB dyB` : tell it where the tube's pointed tip really is, relative to the centre of
  ring m0, in picture pixels, in the left crop (dxA dyA) and in the right crop (dxB dyB). x is to the right, y is
  DOWN. It triangulates the tip, turns the hand so that the TUBE lies on the bore axis and puts the tip in m0. It
  first prints a line with `rays_pass_mm` (should be under 2), `tip_from_tcp_mm` (should be 40 to 52) and
  `tube_vs_bore_axis_deg`. If it prints REJECTED your pixel estimates were inconsistent: look again and retry.
  The same two crops come back. After a fix the cyan line no longer has to follow the tube; what counts is that the
  tube tip is inside m0 and the tube points at m1 in both crops.
- `insert_push [bore]` : pushes 34 mm down the bore axis. Returns one crop from camera B centred on the hand.
- `insert_push_inbore [bore]` : for a start where the tube is already partly inside the bore: pushes to the seat.
- `insert_release` : opens the gripper (checks it opened), backs off 4 cm, waits for the verdict.

Per episode:

1. Start picture + JSON. `nearest_bore.bore` is normally the target bore; check that the hand/tube points at the
   bore marked by the green circle (the three bores used are next to each other; if in doubt use `info`).
2. If `nearest_bore.tip_if_straight_above_opening_mm` is negative AND the picture shows the tube tip inside the
   bore -> `insert_push_inbore`, then go to step 5.
3. Otherwise `insert_align`. If its move says NOT ARRIVED, call `insert_align` again. Then look at both crops:
   is the tube centred on the cyan line with its pointed tip inside ring m0, in BOTH crops?
   - Yes -> step 4.
   - No (the tube runs beside the line, or its tip is outside m0) -> estimate the tip offsets and call `insert_fix`.
     Look at the new crops; one more `insert_fix` is allowed if the tip is still clearly outside m0. Then step 4.
   - If the hand hides the tube tip in one crop, you cannot triangulate: judge from the other crop and go on.
4. `insert_push`.
5. Look at the crop that comes back. Normal: the tube has gone into the bore, only a short piece of tube (under
   about 1.5 cm) shows above the finger pads, `nearest_bore.tcp_above_seated_mm` is about 0 to 4.
   Jam: a long piece of tube sticks out above the fingers and the tube is not in the bore (the hand slid down the
   tube). If it is jammed you may try a recovery of your own with `raw` commands if you see a sound one; otherwise
   go on. Say in your report what you saw.
6. `insert_release`. The reply contains `episode_over`; go on with the next episode.

If an episode gets stuck with no way forward, let it run out:
`raw '{"op":"moves","obs":"none","moves":[{"wait":200}]}'` (it ends as TIMEOUT or FAIL), then continue.
