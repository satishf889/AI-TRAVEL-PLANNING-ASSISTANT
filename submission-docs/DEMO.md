# AI Travel Planning Assistant Demo

Here is a short demo showing RAG, MCP, combined response, and conversational context for the AI Travel Planning Assistant.

## Video Demo
*(Please include a link to the recorded video demo here, e.g., Loom, YouTube, or an MP4 file in the repository.)*

[Link to Demo Video](#)

## Demo Scenarios Covered

1. **RAG (Knowledge Base Retrieval):**
   - The user asks about popular neighbourhoods in Singapore.
   - The assistant retrieves information from the local Markdown documents and cites the source.

2. **MCP Tool Integration (Live Data):**
   - The user asks for the current weather forecast and currency exchange rate.
   - The assistant calls the Open-Meteo and Frankfurter APIs and provides real-time data, clearly labeling the sources.

3. **Combined Response (RAG + MCP + LLM):**
   - The user asks to plan a 2-day itinerary based on a specific budget in USD and the upcoming weather.
   - The assistant converts the currency, checks the weather, and structures an itinerary from the knowledge base, suggesting indoor activities for rainy afternoons.

4. **Conversational Context:**
   - The user follows up with "Can you make it more family-friendly?"
   - The assistant remembers the 2-day itinerary, budget, and weather from the previous turns, and adjusts the activities to include kid-friendly spots like Universal Studios and the S.E.A. Aquarium.
