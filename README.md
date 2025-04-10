# Social Media Simulation Framework

This repository contains a framework for simulating social media dynamics using AI agents with configurable personalities and behaviors. The simulation can be used to study the spread of information, misinformation, and the effects of various moderation strategies.

## Quick Start

### Prerequisites

1. Create a `keys.py` file in the root directory with your OpenAI API key:
```python
OPENAI_API_KEY = "your_OPENAI_API_KEY"
```

2. Install dependencies:
```
pip install -r requirements.txt
```

### Running a Simulation

To run a standard simulation:
```
python src/main.py
```

This will execute a simulation with the configuration defined in `configs/experiment_config.json`. The simulation includes AI agents that interact on a simulated social platform, creating posts, following each other, and interacting with content.

### Configuration Options

The `configs/experiment_config.json` file controls the simulation parameters:

```json
{
    "num_users": 204,              // Number of users in the simulation
    "num_time_steps": 5,           // Number of simulation steps
    "engine": "gpt-4o",            // LLM engine to use (gpt-4o, gpt-3.5-turbo, or local Ollama models)
    "agent_config_generation": "file", 
    "agent_config_path": "personas/personas_from_prolific_description.jsonl",
    "temperature": 1.0,            // Randomness of agent behaviors
    "reset_db": true,              // Whether to reset database on each run
    "initial_follow_probability": 0.1,  // Probability of initial user follows
    "generate_own_post": false,    // Whether agents generate their own posts
    "experiment": {
        "type": "hybrid_fact_checking",  // Type of fact checking: third_party_fact_checking, 
                                         // community_fact_checking, hybrid_fact_checking, no_fact_checking
        "settings": {
            "posts_per_step": 5,   // Number of posts to check per time step
            "fact_checker_temperature": 0.3,  // Temperature for fact checking LLM
            "include_reasoning": true  // Whether to include reasoning in fact check
        }
    }
}
```

## Project Structure

### Simulation Core (`src/`)

- `simulation.py`: Main simulation driver
- `main.py`: Entry point for running a simulation
- `user_manager.py`: Manages user creation and relationships
- `agent_user.py`: Implements user agent behavior
- `news_manager.py`: Manages news injection and tracking
- `fact_checker.py`: Implements fact checking functionality
- `database_manager.py`: Manages SQLite database operations
- `news_spread_analyzer.py`: Analyzes content spread

### Persona Generation (`personas/`)

The `personas/` directory contains tools for generating and managing agent personas:

- `get_personas_from_prolific.py`: Processes demographic data from Prolific surveys to create persona descriptions
- `generate_personas.py`: Generates synthetic user personas
- `personas_from_prolific_description.jsonl`: Contains personas derived from human demographic data

Personas are processed from demographic data and formatted into detailed descriptions that inform agent behavior in the simulation. The simulation can use either synthetic personas or ones derived from real demographic data.

### Experiment Outputs

Simulation results are stored in the `experiment_outputs/` directory:

- `database_copies/`: Contains timestamped SQLite database files from each simulation run
- `configs/`: Stores copies of the configuration used for each run
- `logs/`: Contains simulation logs
- `homophily_analysis/`: Contains homophily analysis results

Each simulation run creates a timestamped database file (e.g., `20250312_083214.db`) that contains all simulation data including users, posts, interactions, and fact checks.

### Post Simulation Analysis (`post_simulation_analysis/`)

The `post_simulation_analysis/` directory contains tools for analyzing simulation results:

- `post_simulation_analysis.py`: Analyzes a single experiment type
- `analysis_combined.py`: Compares results across different experiment types
- `third_party.db`, `community_based.db`, `hybrid.db`, `no_fact_check.db`: Database files for different fact-checking approaches

To run analysis on simulation data:
```
cd post_simulation_analysis
python post_simulation_analysis.py  # Analyzes a single simulation
python analysis_combined.py  # Compares multiple simulation types
```

Analysis results are output as plots in the `plots/` directory.

### Human Study (`human_study/`)

The `human_study/` directory contains data and analysis from a replication study with human participants:

- `demographic_data.csv`: Demographic information of study participants
- `human_study_data.csv`: Responses from human participants
- `formatted_social_feeds.jsonl`: The social media feeds shown to participants
- `plot_demographic_grid_improved.py`: Visualizes demographic distributions
- `get_demographic_summary.py`: Summarizes demographic information

The human study data is used to validate agent behavior and ensure it reflects realistic human responses to social media content.

## Running Other Tools

### Agent Interviews

To interview simulated agents:
```
python src/interview_agents.py --reset
```

Where `--reset` is optional and will delete previous interview records.

## Database Structure

The simulation data is stored in SQLite databases with the following tables:

- `users`: User information and attributes
- `posts`: All posts created in the simulation
- `comments`: Comments on posts
- `follows`: User follow relationships
- `interactions`: User interactions with content
- `spread_metrics`: Metrics tracking content spread
- `fact_checks`: Fact checking verdicts and explanations

You can use any SQLite viewer (like the SQLite viewer in VSCode) to explore the database contents.

## Experimental Conditions

The framework supports several experimental conditions for fact-checking:

1. **No Fact Checking**: Baseline condition with no fact checking
2. **Third-Party Fact Checking**: A centralized fact checker examines content
3. **Community-Based Fact Checking**: Users flag and moderate content
4. **Hybrid Fact Checking**: Combines third-party and community approaches

## Contributing

Please refer to the documentation in individual source files for detailed information on components and their functionality.
