#!/usr/bin/env python3
"""
Music Generation Platform Demo CLI

Interactive demo tool to showcase the complete microservices platform.
Provides easy user experience for testing all features.
"""

import asyncio
import time
from typing import Dict, Optional

import click
import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

console = Console()


class MusicGenClient:
    """Client for interacting with the Music Generation API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))
        self.token: Optional[str] = None
        self.user_info: Optional[Dict] = None
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
        
    async def health_check(self) -> bool:
        """Check if the API is healthy"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except:
            return False
            
    async def register_user(self, username: str, email: str, password: str, full_name: str = "") -> Dict:
        """Register a new user"""
        data = {
            "username": username,
            "email": email,
            "password": password,
            "full_name": full_name
        }
        
        response = await self.client.post(f"{self.base_url}/auth/register", json=data)
        response.raise_for_status()
        
        result = response.json()
        self.token = result["access_token"]
        self.user_info = result["user"]
        
        return result
        
    async def login_user(self, email: str, password: str) -> Dict:
        """Login an existing user"""
        data = {
            "username": email,  # API expects username field for email
            "password": password
        }
        
        response = await self.client.post(
            f"{self.base_url}/auth/login",
            data=data
        )
        response.raise_for_status()
        
        result = response.json()
        self.token = result["access_token"]
        self.user_info = result["user"]
        
        return result
        
    async def generate_music(
        self,
        prompt: str,
        duration: float = 30.0,
        genre: Optional[str] = None,
        mood: Optional[str] = None,
        instruments: Optional[list] = None
    ) -> Dict:
        """Generate music from text prompt"""
        if not self.token:
            raise ValueError("Must be logged in to generate music")
            
        data = {
            "prompt": prompt,
            "duration": duration
        }
        
        if genre:
            data["genre"] = genre
        if mood:
            data["mood"] = mood
        if instruments:
            data["instruments"] = instruments
            
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.post(
            f"{self.base_url}/generate",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def get_job_status(self, job_id: str) -> Dict:
        """Get generation job status"""
        if not self.token:
            raise ValueError("Must be logged in")
            
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.get(
            f"{self.base_url}/generate/job/{job_id}",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def get_profile(self) -> Dict:
        """Get user profile"""
        if not self.token:
            raise ValueError("Must be logged in")
            
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.get(
            f"{self.base_url}/auth/me",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def create_playlist(self, name: str, description: str = "", is_public: bool = True) -> Dict:
        """Create a new playlist"""
        if not self.token:
            raise ValueError("Must be logged in")
            
        data = {
            "name": name,
            "description": description,
            "is_public": is_public,
            "tags": []
        }
        
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.post(
            f"{self.base_url}/playlists",
            json=data,
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def get_playlists(self) -> Dict:
        """Get user's playlists"""
        if not self.token:
            raise ValueError("Must be logged in")
            
        headers = {"Authorization": f"Bearer {self.token}"}
        
        response = await self.client.get(
            f"{self.base_url}/playlists",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json()
        
    async def get_services_health(self) -> Dict:
        """Get health status of all services"""
        response = await self.client.get(f"{self.base_url}/health/services")
        response.raise_for_status()
        
        return response.json()


async def wait_for_generation(client: MusicGenClient, job_id: str) -> Dict:
    """Wait for music generation to complete with progress"""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Generating music...", total=None)
        
        while True:
            try:
                job = await client.get_job_status(job_id)
                
                if job["status"] == "completed":
                    progress.update(task, description="✅ Generation complete!")
                    return job
                elif job["status"] == "failed":
                    progress.update(task, description="❌ Generation failed")
                    return job
                else:
                    progress_percent = job.get("progress", 0)
                    progress.update(
                        task,
                        description=f"Generating... {progress_percent:.1f}%"
                    )
                    
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Generation cancelled by user[/yellow]")
                return {"status": "cancelled"}
            except Exception as e:
                console.print(f"\n[red]Error checking job status: {e}[/red]")
                return {"status": "error", "error": str(e)}


@click.group()
@click.option('--api-url', default='http://localhost:8000', help='API base URL')
@click.pass_context
def cli(ctx, api_url):
    """Music Generation Platform Demo CLI"""
    ctx.ensure_object(dict)
    ctx.obj['api_url'] = api_url


@cli.command()
@click.pass_context
async def setup(ctx):
    """Check system health and setup"""
    api_url = ctx.obj['api_url']
    
    console.print(Panel.fit("🎵 Music Generation Platform Setup", style="bold blue"))
    
    async with MusicGenClient(api_url) as client:
        console.print("\n[bold]Checking system health...[/bold]")
        
        # Check API
        with console.status("Checking API..."):
            if await client.health_check():
                console.print("✅ API is healthy")
            else:
                console.print("❌ API is not responding")
                console.print("Make sure the system is running:")
                console.print("  • For Docker: docker-compose up")
                console.print("  • For local: uvicorn music_gen.api.main:app --reload")
                return
                
        # Check all services
        with console.status("Checking microservices..."):
            try:
                health = await client.get_services_health()
                
                table = Table(title="Service Health Status")
                table.add_column("Service", style="cyan")
                table.add_column("Status", style="magenta")
                table.add_column("Response Time", style="green")
                
                for service_name, service_data in health.get("services", {}).items():
                    status = service_data.get("status", "unknown")
                    response_time = service_data.get("response_time", 0)
                    
                    status_emoji = "✅" if status == "healthy" else "❌"
                    table.add_row(
                        service_name,
                        f"{status_emoji} {status}",
                        f"{response_time:.3f}s" if response_time else "N/A"
                    )
                    
                console.print(table)
                
                healthy_count = health.get("healthy_services", 0)
                total_count = health.get("total_services", 0)
                
                if healthy_count == total_count:
                    console.print(f"\n✅ All {total_count} services are healthy!")
                    console.print("\n[bold green]System is ready for demo![/bold green]")
                else:
                    console.print(f"\n⚠️  {healthy_count}/{total_count} services are healthy")
                    console.print("Some services may not be fully started yet. Try again in a moment.")
                    
            except Exception as e:
                console.print(f"❌ Error checking services: {e}")


@cli.command()
@click.pass_context
async def demo(ctx):
    """Run interactive demo"""
    api_url = ctx.obj['api_url']
    
    console.print(Panel.fit("🎵 Music Generation Platform Demo", style="bold blue"))
    
    async with MusicGenClient(api_url) as client:
        # Check system health first
        if not await client.health_check():
            console.print("❌ System is not running. Please run setup first.")
            return
            
        console.print("\n[bold]Welcome to the Music Generation Platform![/bold]")
        console.print("This demo will show you how to:")
        console.print("• Register a user account")
        console.print("• Generate music from text prompts")
        console.print("• Create and manage playlists")
        console.print("• Explore social features")
        
        # User registration/login
        console.print("\n" + "="*50)
        console.print("[bold]Step 1: User Authentication[/bold]")
        
        has_account = Confirm.ask("Do you already have an account?")
        
        if has_account:
            email = Prompt.ask("Email")
            password = Prompt.ask("Password", password=True)
            
            try:
                with console.status("Logging in..."):
                    result = await client.login_user(email, password)
                console.print(f"✅ Welcome back, {result['user']['username']}!")
            except Exception as e:
                console.print(f"❌ Login failed: {e}")
                return
        else:
            console.print("\nLet's create a new account:")
            username = Prompt.ask("Username")
            email = Prompt.ask("Email")
            password = Prompt.ask("Password", password=True)
            full_name = Prompt.ask("Full Name (optional)", default="")
            
            try:
                with console.status("Creating account..."):
                    result = await client.register_user(username, email, password, full_name)
                console.print(f"✅ Account created! Welcome, {username}!")
            except Exception as e:
                console.print(f"❌ Registration failed: {e}")
                return
                
        # Music Generation
        console.print("\n" + "="*50)
        console.print("[bold]Step 2: Generate Music[/bold]")
        
        sample_prompts = [
            "Upbeat jazz piano with saxophone solo",
            "Relaxing ambient electronic music",
            "Energetic rock guitar with drums",
            "Classical orchestral piece with strings",
            "Lo-fi hip hop beats for studying"
        ]
        
        console.print("\nHere are some sample prompts:")
        for i, prompt in enumerate(sample_prompts, 1):
            console.print(f"{i}. {prompt}")
            
        console.print("\nOr create your own!")
        
        use_sample = Confirm.ask("Use a sample prompt?", default=True)
        
        if use_sample:
            choice = int(Prompt.ask("Choose a prompt (1-5)", default="1")) - 1
            prompt = sample_prompts[choice]
        else:
            prompt = Prompt.ask("Enter your music prompt")
            
        duration = float(Prompt.ask("Duration in seconds", default="30"))
        
        console.print(f"\n🎼 Generating: '{prompt}' ({duration}s)")
        
        try:
            # Start generation
            job = await client.generate_music(prompt, duration)
            job_id = job["job_id"]
            
            console.print(f"Job ID: {job_id}")
            
            # Wait for completion
            result = await wait_for_generation(client, job_id)
            
            if result["status"] == "completed":
                audio_url = result.get("audio_url", "")
                console.print("\n🎉 Music generated successfully!")
                console.print(f"Audio URL: {audio_url}")
                console.print(f"Duration: {result.get('duration_generated', duration)}s")
                
                # Show generation details
                details_panel = Panel(
                    f"[bold]Prompt:[/bold] {prompt}\n"
                    f"[bold]Duration:[/bold] {duration}s\n"
                    f"[bold]Status:[/bold] {result['status']}\n"
                    f"[bold]Audio URL:[/bold] {audio_url}",
                    title="🎵 Generated Music",
                    border_style="green"
                )
                console.print(details_panel)
                
            else:
                console.print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            console.print(f"❌ Generation error: {e}")
            
        # Playlist Management
        console.print("\n" + "="*50)
        console.print("[bold]Step 3: Create a Playlist[/bold]")
        
        if Confirm.ask("Create a playlist for your generated music?"):
            playlist_name = Prompt.ask("Playlist name", default="My Generated Music")
            playlist_desc = Prompt.ask("Description (optional)", default="")
            
            try:
                with console.status("Creating playlist..."):
                    playlist = await client.create_playlist(playlist_name, playlist_desc)
                console.print(f"✅ Playlist '{playlist_name}' created!")
                
                # Show playlists
                playlists = await client.get_playlists()
                if playlists.get("playlists"):
                    table = Table(title="Your Playlists")
                    table.add_column("Name", style="cyan")
                    table.add_column("Tracks", style="magenta")
                    table.add_column("Public", style="green")
                    
                    for pl in playlists["playlists"]:
                        table.add_row(
                            pl["name"],
                            str(pl["track_count"]),
                            "Yes" if pl["is_public"] else "No"
                        )
                    console.print(table)
                    
            except Exception as e:
                console.print(f"❌ Playlist creation failed: {e}")
                
        # User Profile
        console.print("\n" + "="*50)
        console.print("[bold]Step 4: View Your Profile[/bold]")
        
        try:
            with console.status("Fetching profile..."):
                profile = await client.get_profile()
                
            profile_panel = Panel(
                f"[bold]Username:[/bold] {profile['username']}\n"
                f"[bold]Email:[/bold] {profile['email']}\n"
                f"[bold]Tier:[/bold] {profile['tier']}\n"
                f"[bold]Tracks Generated:[/bold] {profile.get('tracks_generated', 0)}\n"
                f"[bold]Playlists:[/bold] {profile.get('playlists_count', 0)}\n"
                f"[bold]Followers:[/bold] {profile.get('followers_count', 0)}",
                title="👤 Your Profile",
                border_style="blue"
            )
            console.print(profile_panel)
            
        except Exception as e:
            console.print(f"❌ Error fetching profile: {e}")
            
        # Demo Complete
        console.print("\n" + "="*50)
        console.print("[bold green]Demo Complete! 🎉[/bold green]")
        console.print("\nYou've successfully:")
        console.print("✅ Created/logged into an account")
        console.print("✅ Generated music from a text prompt")
        console.print("✅ Created a playlist")
        console.print("✅ Viewed your profile")
        
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("• Try more complex prompts with genres and moods")
        console.print("• Generate longer compositions (up to 5 minutes)")
        console.print("• Follow other users and explore social features")
        console.print("• Use the audio processing features for analysis")
        console.print("• Create structured songs with verse-chorus patterns")
        
        console.print(f"\n[dim]API running at: {api_url}[/dim]")


@cli.command()
@click.pass_context
async def quick_test(ctx):
    """Quick system test"""
    api_url = ctx.obj['api_url']
    
    console.print("[bold]Running quick system test...[/bold]")
    
    async with MusicGenClient(api_url) as client:
        try:
            # Test 1: Health check
            console.print("1. Health check...", end=" ")
            if await client.health_check():
                console.print("✅")
            else:
                console.print("❌")
                return
                
            # Test 2: Services health
            console.print("2. Services health...", end=" ")
            health = await client.get_services_health()
            healthy = health.get("healthy_services", 0)
            total = health.get("total_services", 0)
            
            if healthy == total:
                console.print("✅")
            else:
                console.print(f"⚠️  ({healthy}/{total})")
                
            # Test 3: User registration
            console.print("3. User registration...", end=" ")
            test_user = f"testuser_{int(time.time())}"
            try:
                await client.register_user(
                    username=test_user,
                    email=f"{test_user}@example.com",
                    password="testpass123",
                    full_name="Test User"
                )
                console.print("✅")
            except Exception as e:
                console.print(f"❌ {e}")
                return
                
            # Test 4: Music generation
            console.print("4. Music generation...", end=" ")
            try:
                job = await client.generate_music(
                    prompt="Short test melody",
                    duration=10.0
                )
                
                # Quick status check
                status = await client.get_job_status(job["job_id"])
                console.print("✅")
            except Exception as e:
                console.print(f"❌ {e}")
                
            console.print("\n[bold green]Quick test completed![/bold green]")
            
        except Exception as e:
            console.print(f"\n❌ Test failed: {e}")


# CLI command wrappers
def run_async(coro):
    """Run async function in sync context"""
    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))
    return wrapper


# Apply async wrapper to commands
setup.callback = run_async(setup.callback)
demo.callback = run_async(demo.callback)
quick_test.callback = run_async(quick_test.callback)


if __name__ == "__main__":
    cli()