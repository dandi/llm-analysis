# LLM Analysis

This is a repository for setting up Large Language Model (LLM) agents to analyze data on DANDI.

Aim 1 is to use LLMs and open data on DANDI to reproduce key foundational findings in systems neuroscience data analysis.

## Systems neuroscience topics

We have identified 10 key findings in systems neuroscience:

### Visual system

1. Orientation selectivity
2. Linear receptive field modeling

### Auditory system

3. Frequency tuning
4. Spectrotemporal receptive fields

### Motor system

5. Tuning for direction and speed
6. Tuning during reach planning

### Navigational systems

7. Place fields and grid cells
8. Sequential activity and memory replay
9. Theta phase entrainment
10. Theta phase precession

These studies are intended to be foundational in the sense that many current studies rely on and extend these analyses. For example, a more contemporary study might look at how frequency tuning changes in different conditions (experiences, stimulation, maturity, attentional state, etc.). An ability to reproduce these foundational results is therefore necessary to reproduce these more contemporary studies. 

More details about these topics can be found [here](https://docs.google.com/document/d/1_hDuqh_iK4Ali4Kh4Ow5O-ODej91-6F72IbWzZWoSVA).

## Approach

Our initial approach is to use [Cline](https://cline.bot/), a LLM coding agent that is implemented as a Visual Studio Code extension. Cline can receive instructions and read, write, and execute code to accomplish the task. Cline can also use tools that are made available through the [Model Context Protocol (MCP)](https://www.anthropic.com/news/model-context-protocol), which allows us to expand the capabilities of the agent. Cline supports a wide range of LLMs- here, we use Anthropic 3.7 Sonnet with thinking, as it provides the best performance in our experimentation.

### MCPs

For Cline to accomplish this task, its capatibilities need to be augmented with MCP tools. We provide the following tools, defined in [neurosift](https://github.com/flatironinstitute/neurosift):

* dandi_search: Search for datasets in the DANDI Archive using the standard text search feature
* dandi_semantic_search: Semantic search for DANDI datasets using natural language
* dandiset_info: Get detailed information about a specific DANDI dataset including neurodata objects and metadata
* dandiset_assets: List assets/files in a DANDI dataset
* nwb_file_info: Get information about an NWB file including neurodata objects
* dandi_list_neurodata_types: List all unique neurodata types in DANDI archive
* dandi_search_by_neurodata_type: Search for datasets containing specific neurodata types


### Cline instructions

A big part of getting the LLM to do what you want is to have detailed instructions. These instructions can be provided in a .clineinstruct file.

## Steps to reproduce

1. Clone this repository: `git clone https://github.com/dandi/llm-analysis.git`
2. Download and install Visual Studio Code and open this repository within the application.
3. In Visual Studio Code, install the Cline extension.
4. Install neurosift MCPs: https://github.com/flatironinstitute/neurosift/blob/main-v2/docs/mcp-neurosift-tools.md
5. Use `anthropic/claude-3.7-sonnet:thinking` with extended thinking enabled and a Budget of 1,024 tokens.
6. Open Cline and switch to `Plan` mode.
7. Within the prompt, type:
    a. Demonstrate orientation selectivity for neurons in visual areas of the brain.



