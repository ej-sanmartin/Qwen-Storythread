// Projection glossary for the adapter report. The groups, names, and
// descriptions are specific to Qwen3.5-4B (Gated DeltaNet layers mixed with
// full-attention layers) and are used only when the architecture is known.
// Other checkpoints plot with exact names and values but no descriptions.
const GROUPS = {
  linear_attn: "Gated DeltaNet",
  mlp: "Feed-forward",
  self_attn: "Full attention",
};
// [module prefix, projection name, plain-language description]
const GLOSSARY = [
  ["linear_attn", "in_proj_a", "Produces the decay signal that controls how much of the previous recurrent state carries forward."],
  ["linear_attn", "in_proj_b", "Produces the update gate that controls how strongly the current token changes the recurrent state."],
  ["linear_attn", "in_proj_qkv", "Builds the queries, keys, and values used to read from and update the recurrent state."],
  ["linear_attn", "in_proj_z", "Gates the output read from that state before it is projected back into the block."],
  ["linear_attn", "out_proj", "Maps the gated DeltaNet output back to the model\u2019s hidden width."],
  ["mlp", "gate_proj", "Computes a gate that modulates the expanded features."],
  ["mlp", "up_proj", "Expands the token representation into a wider set of features."],
  ["mlp", "down_proj", "Maps the gated features back to the model\u2019s hidden width."],
  ["self_attn", "q_proj", "Builds the queries that determine what the current token looks for. In Qwen3.5 it also produces an attention output gate."],
  ["self_attn", "k_proj", "Builds the keys that queries are matched against."],
  ["self_attn", "v_proj", "Builds the values that attention combines from the tokens it attends to."],
  ["self_attn", "o_proj", "Maps the combined, gated attention output back to the model\u2019s hidden width."],
];
const GLOSSARY_BY_ID = new Map(GLOSSARY.map(([prefix, label, text]) =>
  [`${prefix}.${label}`, {group: GROUPS[prefix], label, text}]));

// Returns the column ids of every glossary row in a group title such as
// "Feed-forward".
function groupColumnIds(group) {
  return GLOSSARY.filter(([prefix]) => GROUPS[prefix] === group)
    .map(([prefix, label]) => `${prefix}.${label}`);
}

// True when the report detected a Qwen3.5 checkpoint from its tensor paths.
const ARCHITECTURE_KNOWN = REPORT.architecture === "qwen3_5";

// Returns {group, label, text} for a column id such as
// "linear_attn.in_proj_a", or null when the architecture is unknown or the
// projection is not in the glossary.
function describe(columnId) {
  return ARCHITECTURE_KNOWN ? GLOSSARY_BY_ID.get(columnId) || null : null;
}
