Modeling and Optimization of Epidemiological Control Policies through Reinforcement Learning

Abstract:
Pandemics involve high transmission of a disease that impact global and local health and economical patterns. The impact of a pandemic can be minimized by enforcing certain restrictions on a community. However, these restrictions, while minimizing infection and death rates, can also lead to economic crisis. Epidemiological models help to propose pandemic control strategies based on non-pharmaceutical interventions such as social distancing, curfews, and lockdowns which reduce the economic impact of these restrictions. However, designing manual control strategies while taking into account disease spread and economical status is non-trivial. Optimal strategies can be designed through multi-objective reinforcement learning (MORL) models which demonstrate how restrictions can be used to optimize the outcome of a pandemic. In this research, I utilize an epidemiological SEIRD (Susceptible, Exposed, Infected, Recovered, Deceased) model – a compartmental model for virtually simulating a pandemic day-by-day. This is combined with a deep double recurrent Q-network to train an agent to enforce the optimal restriction on the SEIRD simulation based on a reward function. I tested two agents with unique reward functions and pandemic goals to obtain two strategies. The first agent placed long lockdowns to reduce the initial spread of the disease, followed by cyclical and shorter lockdowns to mitigate the resurgence of the disease. The second agent provided similar infection rates but an improved economy by implementing a 10-day lockdown and 20-day no restriction cycle. By implementing these automated strategies, humanity can efficiently develop novel pandemic control policies that minimize disease spread and the economic impacts of pandemics.
