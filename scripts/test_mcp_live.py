import sys
import os

# Add the project root to the Python path so we can import features
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from features.mcp.weather_tool import WeatherTool
from features.mcp.currency_tool import CurrencyTool
from features.mcp.mcp_client import MCPClient

def main():
    print("=== Testing Weather Tool ===")
    weather = WeatherTool()
    
    print("\n1. Current Conditions in Singapore:")
    current = weather.get_current_conditions()
    print(f"Date: {current.date}")
    print(f"Description: {current.weather_description}")
    print(f"High: {current.temperature_max_c}°C, Low: {current.temperature_min_c}°C")
    print(f"Rainy? {current.is_rainy}")
    
    print("\n2. 3-Day Forecast:")
    forecast = weather.get_forecast(days=3)
    for day in forecast.days:
        print(f" - {day.date}: {day.weather_description} (Max {day.temperature_max_c}°C)")

    print("\n\n=== Testing Currency Tool ===")
    currency = CurrencyTool()
    
    print("\n1. Live Exchange Rate (INR -> SGD):")
    rate = currency.get_rate("INR", "SGD")
    print(f"1 INR = {rate} SGD")
    
    print("\n2. Conversion (50,000 INR -> SGD):")
    result = currency.convert(50000, "INR", "SGD")
    print(f"{result.amount} {result.from_currency} = {result.converted_amount:.2f} {result.to_currency}")
    
    print("\n\n=== Testing MCP Client Routing ===")
    client = MCPClient(weather_tool=weather, currency_tool=currency)
    
    queries = [
        "Will it rain tomorrow?",
        "Convert 1000 USD to SGD",
        "What are the best places to eat in Singapore?"
    ]
    
    for q in queries:
        selected = client.select_tool(q)
        print(f"Query: '{q}'\n -> Routes to tool: {selected}\n")

if __name__ == "__main__":
    main()
