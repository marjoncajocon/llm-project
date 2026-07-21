// agent.h - Minimal agent loop: parse -> approve -> execute -> observe
#ifndef AGENT_H
#define AGENT_H

#include "approval.h"

// Run the agent loop over a sequence of model responses. Each response is parsed
// for a tool call; approved tool calls are executed and their observations printed
// (in a real agent, observations would be fed back into the model's context). A
// <done> block or a plain-text response ends the loop.
//
// `responses` is the pluggable "model backend" — here a fixed script; a real
// integration would replace it with live LLM calls that see the observations.
// Returns 0 on success.
int agent_run(const char *goal, const char **responses, int n_responses, ApprovalMode mode);

#endif // AGENT_H
