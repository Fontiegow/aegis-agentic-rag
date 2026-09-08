import asyncio
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

API_URL = "http://localhost:8000/api/v1/conversations/chat"
console = Console()

async def main():
    console.print(Panel.fit("[bold cyan]Aegis RAG Agent - Terminal Interface[/bold cyan]", border_style="cyan"))
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        while True:
            try:
                user_input = console.input("\n[bold green]User > [/bold green]").strip()
                if not user_input or user_input.lower() in ["exit", "quit"]:
                    console.print("[yellow]Exiting CLI session.[/yellow]")
                    break

                with console.status("[bold blue]Querying vector database & generating response...[/bold blue]"):
                    response = await client.post(
                        API_URL,
                        json={"prompt": user_input}
                    )
                    response.raise_for_status()
                    data = response.json()

                answer = data.get("response", "No response returned.")
                sources = data.get("sources", [])

                console.print("\n[bold magenta]Aegis Agent >[/bold magenta]")
                console.print(Markdown(answer))

                if sources:
                    console.print("\n[dim]Retrieved Context Sources:[/dim]")
                    for src in sources:
                        console.print(f" - [dim]{src.get('title', 'Doc')}: Score {src.get('score', 0):.3f}[/dim]")

            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[bold red]Error connecting to API:[/bold red] {e}")

if __name__ == "__main__":
    asyncio.run(main())