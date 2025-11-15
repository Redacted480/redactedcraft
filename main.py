#!/usr/bin/env python3
"""Minimal Ursina-based prototype that reads existing block assets and spawns
an explorable flat world. Dependencies are installed on-demand the first time
this script is executed. Runtime activity is logged for easier debugging."""

from __future__ import annotations

import logging
import importlib
import json
import random
import subprocess
import sys
import traceback
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from send2trash import send2trash

from importlib import metadata as importlib_metadata

# TextField patch will be applied at runtime after Ursina is imported

PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PROJECT_ROOT / "assets"
BLOCK_MODEL_PATH = ASSETS_DIR / "blocks" / "Models" / "block.json"
TEXTURES_DIR = ASSETS_DIR / "blocks" / "textures"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "game.log"
WORLD_SAVE_DIR = PROJECT_ROOT / "worlds"
WORLD_METADATA_FILENAME = "world.json"
WORLD_STATE_FILENAME = "blocks.json"
SETTINGS_FILE = PROJECT_ROOT / "settings.json"

DEFAULT_GENERATOR_SETTINGS: Dict[str, Any] = {
    "type": "flat",
    "width": 12,
    "depth": 12,
    "block_size": 1.0,
    "height_variation": 0,
}

DEFAULT_SETTINGS: Dict[str, Any] = {
    "resolution": "auto",  # "auto" or "WIDTHxHEIGHT" (e.g., "1920x1080")
    "refresh_rate": 60,  # Hz
    "fullscreen": False,
    "vsync": True,
    "sound_volume": 1.0,  # 0.0 to 1.0
    "music_volume": 0.7,  # 0.0 to 1.0
    "fov": 90,  # Field of view in degrees (60-120)
}


def slugify_name(value: str) -> str:
    normalized = "".join(
        ch.lower() if ch.isalnum() else "_" if ch in {" ", "-", "_"} else ""
        for ch in value
    )
    slug = "_".join(segment for segment in normalized.split("_") if segment)
    return slug[:48] or "world"


def get_monitor_resolution() -> tuple[int, int]:
    """Get the primary monitor's native/maximum resolution."""
    try:
        import ctypes
        from ctypes import wintypes
        
        # Define DEVMODE structure
        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ('dmDeviceName', ctypes.c_wchar * 32),
                ('dmSpecVersion', ctypes.c_ushort),
                ('dmDriverVersion', ctypes.c_ushort),
                ('dmSize', ctypes.c_ushort),
                ('dmDriverExtra', ctypes.c_ushort),
                ('dmFields', ctypes.c_ulong),
                ('dmPositionX', ctypes.c_long),
                ('dmPositionY', ctypes.c_long),
                ('dmDisplayOrientation', ctypes.c_ulong),
                ('dmDisplayFixedOutput', ctypes.c_ulong),
                ('dmColor', ctypes.c_short),
                ('dmDuplex', ctypes.c_short),
                ('dmYResolution', ctypes.c_short),
                ('dmTTOption', ctypes.c_short),
                ('dmCollate', ctypes.c_short),
                ('dmFormName', ctypes.c_wchar * 32),
                ('dmLogPixels', ctypes.c_ushort),
                ('dmBitsPerPel', ctypes.c_ulong),
                ('dmPelsWidth', ctypes.c_ulong),
                ('dmPelsHeight', ctypes.c_ulong),
                ('dmDisplayFlags', ctypes.c_ulong),
                ('dmDisplayFrequency', ctypes.c_ulong),
                ('dmICMMethod', ctypes.c_ulong),
                ('dmICMIntent', ctypes.c_ulong),
                ('dmMediaType', ctypes.c_ulong),
                ('dmDitherType', ctypes.c_ulong),
                ('dmReserved1', ctypes.c_ulong),
                ('dmReserved2', ctypes.c_ulong),
                ('dmPanningWidth', ctypes.c_ulong),
                ('dmPanningHeight', ctypes.c_ulong),
            ]
        
        # Get all display modes and find the maximum resolution
        user32 = ctypes.windll.user32
        max_width = 0
        max_height = 0
        
        mode_num = 0
        dm = DEVMODE()
        dm.dmSize = ctypes.sizeof(DEVMODE)
        
        # Enumerate all display settings to find the maximum resolution
        while user32.EnumDisplaySettingsW(None, mode_num, ctypes.byref(dm)):
            if dm.dmPelsWidth > max_width or (dm.dmPelsWidth == max_width and dm.dmPelsHeight > max_height):
                max_width = dm.dmPelsWidth
                max_height = dm.dmPelsHeight
            mode_num += 1
        
        if max_width > 0 and max_height > 0:
            return (max_width, max_height)
        else:
            # Fallback to current resolution
            width = user32.GetSystemMetrics(0)
            height = user32.GetSystemMetrics(1)
            return (width, height)
    except Exception as e:
        # Fallback resolution if detection fails
        return (1920, 1080)


def load_settings() -> Dict[str, Any]:
    """Load game settings from file or create with defaults."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                # Merge with defaults to ensure all keys exist
                settings = DEFAULT_SETTINGS.copy()
                settings.update(loaded_settings)
                return settings
        except Exception as e:
            print(f"Error loading settings: {e}. Using defaults.")
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    """Save game settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        print(f"Error saving settings: {e}")


def ensure_unique_world_directory(base_name: str) -> Path:
    candidate = WORLD_SAVE_DIR / base_name
    suffix = 1
    while candidate.exists():
        candidate = WORLD_SAVE_DIR / f"{base_name}_{suffix}"
        suffix += 1
    return candidate


def default_world_metadata(display_name: str, slug: str, mode: str = "creative") -> Dict[str, Any]:
    seed = random.randint(0, 2**31 - 1)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    return {
        "display_name": display_name,
        "slug": slug,
        "seed": seed,
        "created_at": timestamp,
        "last_played_at": timestamp,
        "mode": mode,
        "generator": dict(DEFAULT_GENERATOR_SETTINGS),
    }


def load_world_metadata(world_dir: Path) -> Dict[str, Any]:
    metadata_path = world_dir / WORLD_METADATA_FILENAME
    metadata: Dict[str, Any]
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    else:
        metadata = default_world_metadata(world_dir.name, world_dir.name)
    metadata.setdefault("display_name", world_dir.name)
    metadata.setdefault("slug", world_dir.name)
    metadata.setdefault("mode", "creative")
    metadata.setdefault("generator", dict(DEFAULT_GENERATOR_SETTINGS))
    metadata.setdefault("seed", random.randint(0, 2**31 - 1))
    metadata.setdefault("last_played_at", datetime.utcnow().isoformat(timespec="seconds"))
    return metadata


def save_world_metadata(world_dir: Path, metadata: Dict[str, Any]) -> None:
    metadata_path = world_dir / WORLD_METADATA_FILENAME
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def create_new_world(display_name: str, mode: str = "creative") -> Dict[str, Any]:
    slug = slugify_name(display_name)
    world_dir = ensure_unique_world_directory(slug)
    world_dir.mkdir(parents=True, exist_ok=True)
    metadata = default_world_metadata(display_name, world_dir.name, mode)
    save_world_metadata(world_dir, metadata)
    return {"display_name": metadata["display_name"], "path": world_dir, "metadata": metadata}


def list_worlds() -> list[Dict[str, Any]]:
    worlds: list[Dict[str, Any]] = []
    if not WORLD_SAVE_DIR.exists():
        return worlds
    for path in WORLD_SAVE_DIR.glob("*"):
        if not path.is_dir():
            continue
        metadata = load_world_metadata(path)
        worlds.append(
            {
                "display_name": metadata.get("display_name", path.name),
                "path": path,
                "metadata": metadata,
            }
        )
    worlds.sort(key=lambda info: info["metadata"].get("last_played_at", ""), reverse=True)
    return worlds

REQUIRED_PACKAGES: Dict[str, Dict[str, Optional[str]]] = {
    "ursina": {"version": "5.0.0", "module": "ursina"},
    "send2trash": {"version": None, "module": "send2trash"},
}

logger = logging.getLogger("redactedcraft")

# Apply runtime patch to fix TextField selection bug
def fix_text_field_selection_bug():
    """Fix the TextField selection bug by monkey-patching the class"""
    try:
        # Import only after Ursina is loaded
        from ursina.prefabs.text_field import TextField
        from ursina import Entity, destroy
        
        # Create a patched version of the draw_selection method
        def patched_draw_selection(self):
            try:
                for e in self.selection_parent.children:
                    destroy(e)

                if self.selection[0] == self.selection[1]:
                    return

                sel = self._ordered_selection()
                
                # Handle both object and tuple formats for selection
                try:
                    # Try as objects first (normal case)
                    start_y = int(sel[0].y)
                    end_y = int(sel[1].y)
                    start_x = sel[0].x
                    end_x = sel[1].x
                except (AttributeError, TypeError):
                    # Fall back to tuple indices (when it would crash)
                    start_y = int(sel[0][1]) if isinstance(sel[0], tuple) else 0
                    end_y = int(sel[1][1]) if isinstance(sel[1], tuple) else 0
                    start_x = sel[0][0] if isinstance(sel[0], tuple) else 0
                    end_x = sel[1][0] if isinstance(sel[1], tuple) else 0
                    
                lines = self.text.split('\n')
                
                try:
                    # Create highlight for selection
                    if start_y == end_y:
                        e = Entity(parent=self.selection_parent, model='cube', origin=(-.5,-.5), 
                                  color=self.highlight_color, ignore=True, y=start_y)
                        e.x = start_x
                        e.scale_x = end_x - start_x
                        return

                    # first line
                    if start_y >= self.scroll and start_y < self.scroll+self.max_lines:
                        e = Entity(parent=self.selection_parent, model='quad', origin=(-.5, -.5),
                                  color=self.highlight_color, double_sided=True, position=(start_x, start_y), ignore=True)
                        e.scale_x = len(lines[start_y]) - start_x

                    # middle lines
                    for y in range(max(start_y+1, self.scroll), min(end_y, self.scroll+self.max_lines)):
                        if y < len(lines):
                            e = Entity(parent=self.selection_parent, model='quad', origin=(-.5, -.5),
                                       color=self.highlight_color, double_sided=True, position=(0,y), ignore=True)
                            e.scale_x = len(lines[y])

                    # last line
                    if end_y >= self.scroll and end_y < self.scroll+self.max_lines and end_y < len(lines):
                        e = Entity(parent=self.selection_parent, model='quad', origin=(-.5, -.5),
                                   color=self.highlight_color, double_sided=True, position=(0,end_y), ignore=True)
                        e.scale_x = end_x
                except Exception as e:
                    logger.debug(f"Error highlighting selection: {e}")
                    # Continue without crashing
            except Exception as e:
                logger.debug(f"Selection error: {e}")
                # Don't crash on selection errors
        
        # Apply the patch
        TextField.draw_selection = patched_draw_selection
        logger.info("Applied TextField selection patch")
        return True
    except Exception as e:
        logger.error(f"Failed to apply TextField patch: {e}")
        return False


def ensure_dependencies() -> None:
    """Install required pip packages if they are unavailable or outdated."""
    
    # Make sure pip is up-to-date first
    try:
        logger.info("Ensuring pip is up to date")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,  # Suppress output
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        logger.warning("Failed to upgrade pip, will continue with existing version")
    
    missing_packages = []
    outdated_packages = []

    # First pass: check what needs to be installed
    for package, meta in REQUIRED_PACKAGES.items():
        module_name = meta.get("module") or package.replace("-", "_")
        version = meta.get("version")
        
        try:
            importlib.import_module(module_name)  # type: ignore[arg-type]
            if version:
                try:
                    installed_version = importlib_metadata.version(package)
                    if installed_version != version:
                        logger.info(
                            "Dependency %s has version %s; expected %s. Will reinstall.",
                            package,
                            installed_version,
                            version,
                        )
                        outdated_packages.append((package, version))
                except (importlib_metadata.PackageNotFoundError, AttributeError):
                    # Package exists but can't determine version - mark for reinstall
                    missing_packages.append((package, version))
        except ImportError:
            missing_packages.append((package, version))
    
    # Install all missing packages in one batch if possible
    if missing_packages:
        logger.info("Installing missing dependencies: %s", ", ".join([p[0] for p in missing_packages]))
        for package, version in missing_packages:
            spec = f"{package}=={version}" if version else package
            try:
                logger.info("Installing %s", spec)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", spec],
                    stdout=subprocess.PIPE,  # Capture output but don't display
                    stderr=subprocess.STDOUT,
                )
                logger.info("Successfully installed %s", spec)
            except subprocess.CalledProcessError as exc:
                logger.error(f"Failed to install {spec}: {exc}")
                # Try to continue with other packages
    
    # Update outdated packages
    if outdated_packages:
        logger.info("Updating outdated dependencies")
        for package, version in outdated_packages:
            spec = f"{package}=={version}" if version else package
            try:
                logger.info("Updating %s", spec)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "--upgrade", spec],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                logger.info("Successfully updated %s", spec)
            except subprocess.CalledProcessError as exc:
                logger.error(f"Failed to update {spec}: {exc}")
                # Try to continue with other packages
    
    # Final verification
    all_ok = True
    for package, meta in REQUIRED_PACKAGES.items():
        module_name = meta.get("module") or package.replace("-", "_")
        try:
            importlib.import_module(module_name)
            logger.info(f"Verified {package} is installed and importable")
        except ImportError:
            logger.error(f"Critical dependency {package} could not be imported after installation attempts")
            all_ok = False
    
    if not all_ok:
        logger.error("Some dependencies could not be installed. The game may not function correctly.")
        # Continue anyway - let user decide whether to proceed with missing deps


def load_block_display_settings(block_type: str) -> Optional[Dict[str, Any]]:
    """Load display.gui settings from block JSON file."""
    json_path = ASSETS_DIR / "blocks" / "Models" / f"{block_type}_block.json"
    if not json_path.exists():
        logger.warning("Block definition not found: %s", json_path)
        return None
    
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            display = data.get("display", {})
            gui_display = display.get("gui", {})
            if gui_display:
                logger.info("Loaded GUI display settings for %s: %s", block_type, gui_display)
                return gui_display
            return None
    except Exception as exc:
        logger.warning("Failed to load display settings for %s: %s", block_type, exc)
        return None


def load_block_definition(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Block definition not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_texture_path(file_hint: Optional[str]) -> Optional[str]:
    if not file_hint:
        return None

    normalized = file_hint.replace("\\", "/").lstrip("./")
    candidates = []
    if normalized.startswith("blocks/"):
        candidates.append(normalized)
    else:
        candidates.extend(
            [
                normalized,
                f"{normalized}.png",
                f"blocks/textures/{normalized}",
                f"blocks/textures/{normalized}.png",
            ]
        )

    for candidate in candidates:
        candidate_path = (ASSETS_DIR / candidate).resolve()
        try:
            candidate_path.relative_to(ASSETS_DIR)
        except ValueError:
            continue
        if candidate_path.exists():
            return candidate_path.relative_to(ASSETS_DIR).as_posix()

    search_root = TEXTURES_DIR
    pattern = Path(normalized).stem
    matches = list(search_root.glob(f"**/{pattern}.*"))
    if matches:
        return matches[0].relative_to(ASSETS_DIR).as_posix()
    return None


def resolve_face_texture_paths(model_data: Dict) -> Dict[str, Optional[str]]:
    textures_map = model_data.get("textures", {})
    elements = model_data.get("elements") or []
    if not elements:
        raise ValueError("Block definition has no elements to build a mesh.")
    faces = elements[0].get("faces", {})

    fallback_map = {
        "north": "blocks/textures/Grass_side.png",
        "south": "blocks/textures/Grass_side.png",
        "east": "blocks/textures/Grass_side.png",
        "west": "blocks/textures/Grass_side.png",
        "up": "blocks/textures/Grass_top.png",
        "down": "blocks/textures/Grass_bottom.png",
    }

    resolved: Dict[str, Optional[str]] = {}
    for face_name in ("north", "south", "east", "west", "up", "down"):
        face_info = faces.get(face_name, {})
        texture_ref = face_info.get("texture")
        file_hint: Optional[str] = None
        if isinstance(texture_ref, str) and texture_ref:
            reference = texture_ref[1:] if texture_ref.startswith("#") else texture_ref
            file_hint = textures_map.get(reference, reference)

        candidate = find_texture_path(file_hint) if file_hint else None
        if candidate is None:
            candidate = fallback_map.get(face_name)
            if candidate and not (ASSETS_DIR / candidate).exists():
                candidate = find_texture_path(candidate)
        resolved[face_name] = candidate
    return resolved


def ensure_directories() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        LOG_DIR,
        WORLD_SAVE_DIR,
        TEXTURES_DIR,
        ASSETS_DIR / "menu" / "backgrounds",
        ASSETS_DIR / "menu" / "buttons",
        ASSETS_DIR / "menu" / "GUIS",
        ASSETS_DIR / "blocks" / "Models",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("All required directories verified/created")


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.info("KeyboardInterrupt: exiting application")
            return
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    sys.excepthook = handle_exception


def main() -> None:
    setup_logging()
    logger.info("Launching RedactedCraft Python prototype")
    ensure_directories()

    ensure_dependencies()
    logger.info("Dependencies verified")

    # Imports that rely on the installed dependencies happen after ensuring they exist.
    from ursina import (  # type: ignore[import]
        Ursina,
        Button,
        Entity,
        InputField,
        Sky,
        Slider,
        Text,
        Vec2,
        Vec3,
        application,
        camera,
        color,
        destroy,
        held_keys,
        invoke,
        mouse,
        load_texture,
        raycast,
        BoxCollider,
        scene,
        time,
        window,
        Quad,
    )
    from ursina.prefabs.first_person_controller import FirstPersonController  # type: ignore[import]

    application.asset_folder = ASSETS_DIR
    application.compressed_textures_folder = ASSETS_DIR

    # Performance optimizations for Ursina/Panda3D
    from panda3d.core import loadPrcFileData
    
    # GPU and rendering optimizations
    loadPrcFileData('', 'framebuffer-hardware true')  # Use GPU for framebuffer
    loadPrcFileData('', 'framebuffer-multisample true')  # Enable MSAA
    loadPrcFileData('', 'multisamples 4')  # 4x MSAA for quality
    loadPrcFileData('', 'texture-anisotropic-degree 16')  # Better texture filtering
    loadPrcFileData('', 'sync-video false')  # Disable V-Sync for higher FPS
    loadPrcFileData('', 'show-frame-rate-meter false')  # Clean display
    # Keep Panda3D's default Z-up coordinate system for correct texture orientation
    loadPrcFileData('', 'threading-model Cull/Draw')  # Multi-threaded rendering
    
    # Load game settings
    game_settings = load_settings()
    
    # Save settings file if it doesn't exist (creates default settings.json)
    if not SETTINGS_FILE.exists():
        save_settings(game_settings)
        logger.info("Created default settings file")
    
    # Determine resolution
    if game_settings["resolution"] == "auto":
        monitor_width, monitor_height = get_monitor_resolution()
        logger.info(f"Detected monitor resolution: {monitor_width}x{monitor_height}")
        # Use 80% of monitor resolution for windowed mode if not fullscreen
        if not game_settings["fullscreen"]:
            window_width = int(monitor_width * 0.8)
            window_height = int(monitor_height * 0.8)
        else:
            window_width = monitor_width
            window_height = monitor_height
    else:
        # Parse resolution string like "1920x1080"
        try:
            window_width, window_height = map(int, game_settings["resolution"].split("x"))
        except:
            window_width, window_height = 1920, 1080
    
    logger.info(f"Window resolution: {window_width}x{window_height}, Fullscreen: {game_settings['fullscreen']}")
    
    # Set window size in Panda3D config before Ursina initialization
    loadPrcFileData('', f'win-size {window_width} {window_height}')
    loadPrcFileData('', 'win-fixed-size true')  # Prevent window resizing at OS level
    
    app = Ursina(
        borderless=False,
        fullscreen=game_settings["fullscreen"],
        development_mode=True,
        vsync=game_settings["vsync"],
        forced_aspect_ratio=None  # Let window handle aspect ratio
    )
    
    # Prevent window resizing (both Ursina and Panda3D methods)
    window.resizable = False
    if hasattr(window, 'entity'):
        from panda3d.core import WindowProperties
        props = WindowProperties()
        props.setFixedSize(True)
        window.entity.requestProperties(props)
    
    # Update aspect ratio to match actual window
    window.aspect_ratio = window_width / window_height
    
    # Ensure mouse is visible and working in UI
    mouse.locked = False
    mouse.visible = True
    
    window.color = color.rgb(135, 206, 235)
    window.title = "RedactedCraft Python Prototype"
    window.fps_counter.enabled = True  # Show FPS counter
    window.fps_counter.scale = 1.5
    window.fps_counter.position = window.top_right
    
    # Apply FOV setting
    camera.fov = game_settings["fov"]
    
    # Apply fix for TextField selection bug
    fix_text_field_selection_bug()
    
    # Disable F5 hot reload to prevent crashes
    application.hot_reloader.enabled = False

    Sky()

    try:
        block_data = load_block_definition(BLOCK_MODEL_PATH)
    except Exception as exc:
        logger.exception("Unable to read block definition: %s", exc)
        sys.exit(1)

    face_paths = resolve_face_texture_paths(block_data)
    
    logger.info("Resolved texture paths: %s", face_paths)

    missing_faces = [face for face, rel in face_paths.items() if not rel]
    if missing_faces:
        logger.warning(
            "One or more faces did not resolve to a texture. Fallback art will be used for: %s",
            ", ".join(missing_faces),
        )

    # Create face textures - ensure they're loaded without alpha blending
    face_textures = {}
    for face, rel_path in face_paths.items():
        if rel_path:
            texture = load_texture(rel_path)
            if texture:
                logger.info("Loaded texture for %s: %s", face, rel_path)
                # Disable alpha blending for grass textures to make them solid
                texture.filtering = False  # Disable filtering for crisp pixel art
                face_textures[face] = texture
            else:
                logger.warning("Failed to load texture for %s: %s", face, rel_path)
                face_textures[face] = None
        else:
            logger.warning("No texture path for face: %s", face)
            face_textures[face] = None

    face_transforms = {
        "north": {"position": Vec3(0, 0, -0.5), "rotation": Vec3(0, 0, 0)},
        "south": {"position": Vec3(0, 0, 0.5), "rotation": Vec3(0, 180, 0)},
        "east": {"position": Vec3(0.5, 0, 0), "rotation": Vec3(0, 90, 0)},
        "west": {"position": Vec3(-0.5, 0, 0), "rotation": Vec3(0, -90, 0)},
        "up": {"position": Vec3(0, 0.5, 0), "rotation": Vec3(-90, 0, 0)},
        "down": {"position": Vec3(0, -0.5, 0), "rotation": Vec3(90, 0, 0)},
    }

    # Load player model configuration
    PLAYER_MODEL_PATH = ASSETS_DIR / "Models" / "player.cfg"
    player_model_config = None
    try:
        if PLAYER_MODEL_PATH.exists():
            with PLAYER_MODEL_PATH.open("r", encoding="utf-8") as f:
                player_model_config = json.load(f)
            logger.info("Loaded player model configuration")
        else:
            logger.warning("Player model not found at %s", PLAYER_MODEL_PATH)
    except Exception as exc:
        logger.exception("Failed to load player model: %s", exc)

    def create_player_body(parent_entity) -> Dict[str, Entity]:
        """Create Minecraft Steve-style player model.
        Standard Minecraft dimensions: Head 8x8x8, Body 8x12x4, Arms 4x12x4, Legs 4x12x4
        Total height: 32 pixels = 2 blocks"""
        
        # Conversion: 16 pixels = 1 block in Minecraft
        pixel_to_unit = 1.0 / 16.0
        
        body_parts = {}
        
        # Minecraft Steve color scheme
        skin_color = color.rgb(220, 180, 140)  # Skin tone
        shirt_color = color.rgb(75, 135, 180)  # Light blue shirt
        pants_color = color.rgb(45, 60, 135)  # Dark blue pants
        
        # HEAD: 8x8x8 pixels, positioned at top
        # Y position: legs(12) + body(12) + half_head(4) = 28 pixels from feet
        head = Entity(
            parent=parent_entity,
            model='cube',
            scale=(8 * pixel_to_unit, 8 * pixel_to_unit, 8 * pixel_to_unit),
            position=(0, 28 * pixel_to_unit, 0),
            color=skin_color,
            collider=None,
        )
        body_parts['head'] = head
        
        # BODY/TORSO: 8x12x4 pixels (width x height x depth)
        # Y position: legs(12) + half_body(6) = 18 pixels from feet
        body = Entity(
            parent=parent_entity,
            model='cube',
            scale=(8 * pixel_to_unit, 12 * pixel_to_unit, 4 * pixel_to_unit),
            position=(0, 18 * pixel_to_unit, 0),
            color=shirt_color,
            collider=None,
        )
        body_parts['body'] = body
        
        # LEFT ARM: 4x12x4 pixels
        # X position: half body width (4) + half arm width (2) = 6 pixels from center
        # Y position: legs(12) + body(12) = 24 pixels from feet (at shoulder)
        left_arm = Entity(
            parent=parent_entity,
            model='cube',
            scale=(4 * pixel_to_unit, 12 * pixel_to_unit, 4 * pixel_to_unit),
            position=(-6 * pixel_to_unit, 24 * pixel_to_unit, 0),
            color=skin_color,
            collider=None,
            origin=(0, 0.5, 0),  # Pivot at top (shoulder)
        )
        body_parts['leftArm'] = left_arm
        
        # RIGHT ARM: 4x12x4 pixels
        right_arm = Entity(
            parent=parent_entity,
            model='cube',
            scale=(4 * pixel_to_unit, 12 * pixel_to_unit, 4 * pixel_to_unit),
            position=(6 * pixel_to_unit, 24 * pixel_to_unit, 0),
            color=skin_color,
            collider=None,
            origin=(0, 0.5, 0),  # Pivot at top (shoulder)
        )
        body_parts['rightArm'] = right_arm
        
        # LEFT LEG: 4x12x4 pixels
        # X position: 2 pixels from center (half leg width)
        # Y position: 12 pixels from feet (at hip)
        left_leg = Entity(
            parent=parent_entity,
            model='cube',
            scale=(4 * pixel_to_unit, 12 * pixel_to_unit, 4 * pixel_to_unit),
            position=(-2 * pixel_to_unit, 12 * pixel_to_unit, 0),
            color=pants_color,
            collider=None,
            origin=(0, 0.5, 0),  # Pivot at top (hip)
        )
        body_parts['leftLeg'] = left_leg
        
        # RIGHT LEG: 4x12x4 pixels
        right_leg = Entity(
            parent=parent_entity,
            model='cube',
            scale=(4 * pixel_to_unit, 12 * pixel_to_unit, 4 * pixel_to_unit),
            position=(2 * pixel_to_unit, 12 * pixel_to_unit, 0),
            color=pants_color,
            collider=None,
            origin=(0, 0.5, 0),  # Pivot at top (hip)
        )
        body_parts['rightLeg'] = right_leg
        
        return body_parts

    block_type_textures: Dict[str, Dict[str, object]] = {
        "grass": face_textures,
    }

    world_entities: list[Entity] = []
    player_entity = None  # type: Optional[Entity]
    game_state: Dict[str, bool] = {"paused": False}
    pause_menu: Optional["PauseMenu"] = None
    current_world_info: Optional[Dict[str, Any]] = None
    current_world_blocks: Dict[str, str] = {}
    current_player_position: Optional[Vec3] = None
    current_player_orientation: Optional[tuple[float, float]] = None  # yaw, pitch
    current_hotbar_slot: int = 0  # Selected hotbar slot (0-5)
    world_dirty = False
    enable_face_culling = False  # Enable after world generation completes

    def pos_to_key(position: Vec3) -> str:
        """Convert a block world position to a stable string key."""
        return f"{position.x:.2f},{position.y:.2f},{position.z:.2f}"

    def key_to_pos(key: str) -> Vec3:
        """Convert a string key back into a Vec3 position."""
        x_str, y_str, z_str = key.split(",")
        return Vec3(float(x_str), float(y_str), float(z_str))

    def mark_world_dirty() -> None:
        nonlocal world_dirty
        world_dirty = True

    def reset_dirty_flag() -> None:
        nonlocal world_dirty
        world_dirty = False

    def add_block_to_state(block: "Block", *, mark_dirty: bool = True) -> None:
        key = pos_to_key(block.position)
        block.grid_key = key
        current_world_blocks[key] = block.block_type
        if mark_dirty:
            mark_world_dirty()

    def remove_block_from_state(block: "Block", *, mark_dirty: bool = True) -> None:
        key = getattr(block, "grid_key", pos_to_key(block.position))
        if key in current_world_blocks:
            current_world_blocks.pop(key, None)
            if mark_dirty:
                mark_world_dirty()

    def get_opposite_face(face_name: str) -> str:
        """Get the opposite face name (e.g., north -> south)."""
        opposites = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east',
            'up': 'down',
            'down': 'up',
        }
        return opposites.get(face_name, face_name)

    def add_face_to_block(block: "Block", face_name: str) -> None:
        """Add a face to an existing block if it doesn't already exist."""
        # Check if face already exists
        if face_name in block.face_entities:
            return
        
        # Get texture and transform for this face
        texture = block.textures.get(face_name)
        if not texture:
            return
        
        transform = face_transforms.get(face_name)
        if not transform:
            return
        
        # Create the face quad
        face_quad = Entity(
            parent=block,
            model='quad',
            texture=texture,
            color=color.white,
            double_sided=True,
            position=transform['position'],
            rotation=transform['rotation'],
            scale=1,
            origin=(0, 0),
        )
        
        # Configure rendering
        face_quad.set_light_off()
        
        # Optimize texture filtering
        if hasattr(face_quad.texture, 'setMinfilter'):
            from panda3d.core import SamplerState
            face_quad.texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
            face_quad.texture.setMagfilter(SamplerState.FT_linear)
        
        # Apply texture flipping
        if face_name == 'north':
            face_quad.texture_scale = (1, 1)
        elif face_name == 'south':
            face_quad.texture_scale = (-1, 1)
        elif face_name == 'east':
            face_quad.texture_scale = (-1, 1)
        elif face_name == 'west':
            face_quad.texture_scale = (1, 1)
        elif face_name == 'up':
            face_quad.texture_scale = (1, 1)
        elif face_name == 'down':
            face_quad.texture_scale = (1, -1)
        
        # Store reference
        block.face_entities[face_name] = face_quad

    def remove_face_from_block(block: "Block", face_name: str) -> None:
        """Remove a face from an existing block if it exists."""
        if face_name in block.face_entities:
            face_quad = block.face_entities[face_name]
            destroy(face_quad)
            del block.face_entities[face_name]

    def update_adjacent_blocks_on_removal(block_position: Vec3) -> None:
        """Update adjacent blocks to show newly exposed faces when a block is removed."""
        # Define face offsets and their corresponding face names
        face_offsets = {
            'north': (Vec3(0, 0, -1), 'south'),  # Block to the north needs its south face shown
            'south': (Vec3(0, 0, 1), 'north'),   # Block to the south needs its north face shown
            'east': (Vec3(1, 0, 0), 'west'),     # Block to the east needs its west face shown
            'west': (Vec3(-1, 0, 0), 'east'),    # Block to the west needs its east face shown
            'up': (Vec3(0, 1, 0), 'down'),       # Block above needs its down face shown
            'down': (Vec3(0, -1, 0), 'up'),      # Block below needs its up face shown
        }
        
        # Check each adjacent position
        for direction, (offset, face_to_add) in face_offsets.items():
            adjacent_pos = block_position + offset
            adjacent_key = pos_to_key(adjacent_pos)
            
            # Check if there's a block at the adjacent position
            if adjacent_key in current_world_blocks:
                # Find the actual block entity
                for entity in world_entities:
                    if isinstance(entity, Block) and entity.grid_key == adjacent_key:
                        # Add the newly exposed face to this block
                        add_face_to_block(entity, face_to_add)
                        break

    def update_adjacent_blocks_on_placement(block_position: Vec3) -> None:
        """Update adjacent blocks to hide covered faces when a block is placed."""
        # Define face offsets and their corresponding face names
        face_offsets = {
            'north': (Vec3(0, 0, -1), 'south'),  # Block to the north needs its south face hidden
            'south': (Vec3(0, 0, 1), 'north'),   # Block to the south needs its north face hidden
            'east': (Vec3(1, 0, 0), 'west'),     # Block to the east needs its west face hidden
            'west': (Vec3(-1, 0, 0), 'east'),    # Block to the west needs its east face hidden
            'up': (Vec3(0, 1, 0), 'down'),       # Block above needs its down face hidden
            'down': (Vec3(0, -1, 0), 'up'),      # Block below needs its up face hidden
        }
        
        # Check each adjacent position
        for direction, (offset, face_to_remove) in face_offsets.items():
            adjacent_pos = block_position + offset
            adjacent_key = pos_to_key(adjacent_pos)
            
            # Check if there's a block at the adjacent position
            if adjacent_key in current_world_blocks:
                # Find the actual block entity
                for entity in world_entities:
                    if isinstance(entity, Block) and entity.grid_key == adjacent_key:
                        # Remove the now-hidden face from this block
                        remove_face_from_block(entity, face_to_remove)
                        break

    def load_world_state(world_path: Path) -> tuple[Dict[str, str], Optional[Vec3], Optional[tuple[float, float]], int]:
        state_path = world_path / WORLD_STATE_FILENAME
        if not state_path.exists():
            return {}, None, None, 0
        try:
            with state_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # noqa: BLE001 - log and recover
            logger.exception("Failed to load world state from %s: %s", state_path, exc)
            return {}, None, None, 0

        if not isinstance(data, dict):
            logger.warning("Unexpected world state format in %s", state_path)
            return {}, None, None, 0

        blocks_data = data.get("blocks", {})
        block_map: Dict[str, str] = {}

        if isinstance(blocks_data, dict):
            for key, block_type in blocks_data.items():
                if isinstance(key, str) and isinstance(block_type, str):
                    block_map[key] = block_type
        elif isinstance(blocks_data, list):
            for entry in blocks_data:
                if not isinstance(entry, dict):
                    continue
                key = entry.get("pos") or entry.get("position")
                block_type = entry.get("type")
                if isinstance(key, str) and isinstance(block_type, str):
                    block_map[key] = block_type
        else:
            logger.warning("Unsupported block list format in %s", state_path)

        player_pos = data.get("player_position")
        if isinstance(player_pos, dict):
            try:
                x = float(player_pos.get("x", 0.0))
                y = float(player_pos.get("y", 0.0))
                z = float(player_pos.get("z", 0.0))
                position_vec = Vec3(x, y, z)
            except (TypeError, ValueError):
                logger.warning("Invalid player position found in %s", state_path)
                position_vec = None
        else:
            position_vec = None

        orientation_data = data.get("player_orientation")
        orientation: Optional[tuple[float, float]] = None
        if isinstance(orientation_data, dict):
            try:
                yaw = float(orientation_data.get("yaw", 0.0))
                pitch = float(orientation_data.get("pitch", 0.0))
                orientation = (yaw, pitch)
            except (TypeError, ValueError):
                logger.warning("Invalid player orientation found in %s", state_path)

        hotbar_slot = int(data.get("hotbar_slot", 0))
        if not (0 <= hotbar_slot <= 5):
            hotbar_slot = 0

        return block_map, position_vec, orientation, hotbar_slot

    def save_world_state(
        world_path: Path,
        block_map: Dict[str, str],
        player_pos: Optional[Vec3],
        player_orientation: Optional[tuple[float, float]],
        hotbar_slot: int,
    ) -> None:
        state_path = world_path / WORLD_STATE_FILENAME
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {"version": 4, "blocks": block_map, "hotbar_slot": hotbar_slot}
        if player_pos is not None:
            payload["player_position"] = {"x": player_pos.x, "y": player_pos.y, "z": player_pos.z}
        if player_orientation is not None:
            yaw, pitch = player_orientation
            payload["player_orientation"] = {"yaw": yaw, "pitch": pitch}
        try:
            with state_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception as exc:  # noqa: BLE001 - log and propagate
            logger.exception("Failed to save world state to %s: %s", state_path, exc)
            raise

    def save_current_world_state(*, force: bool = False) -> None:
        nonlocal current_player_position, current_player_orientation, current_hotbar_slot
        if current_world_info is None:
            return

        world_path = current_world_info.get("path")
        if world_path is None:
            logger.warning("Current world info missing path; skipping save")
            return

        if not isinstance(world_path, Path):
            world_path = Path(world_path)
            current_world_info["path"] = world_path

        if not force and not world_dirty:
            return

        try:
            player_pos = None
            if player_entity is not None:
                player_pos = Vec3(player_entity.x, player_entity.y, player_entity.z)
            elif current_player_position is not None:
                player_pos = current_player_position

            player_orientation = None
            if player_entity is not None:
                yaw = float(getattr(player_entity, "rotation_y", 0.0))
                pitch = 0.0
                camera_pivot = getattr(player_entity, "camera_pivot", None)
                if camera_pivot is not None:
                    pitch = float(getattr(camera_pivot, "rotation_x", 0.0))
                else:
                    pitch = float(getattr(player_entity, "rotation_x", 0.0))
                player_orientation = (yaw, pitch)
            elif current_player_orientation is not None:
                player_orientation = current_player_orientation

            hotbar_slot = current_hotbar_slot
            if player_entity is not None and hasattr(player_entity, 'selected_slot'):
                hotbar_slot = player_entity.selected_slot
                current_hotbar_slot = hotbar_slot

            save_world_state(world_path, dict(current_world_blocks), player_pos, player_orientation, hotbar_slot)
            logger.info(
                "Saved world state to %s (%d blocks)%s%s",
                world_path,
                len(current_world_blocks),
                f" and player position {player_pos}" if player_pos is not None else "",
                f" with orientation (yaw={player_orientation[0]:.2f}, pitch={player_orientation[1]:.2f})"
                if player_orientation is not None
                else "",
            )
            if player_pos is not None:
                current_player_position = player_pos
            if player_orientation is not None:
                current_player_orientation = player_orientation
            reset_dirty_flag()
        except Exception:
            # save_world_state already logged the failure; leave dirty flag for retry.
            logger.error("Unable to persist world state for %s", world_path)

    def pause_game() -> None:
        if game_state["paused"]:
            return
        logger.info("Pausing game")
        game_state["paused"] = True
        mouse.locked = False
        mouse.visible = True
        logger.info(f"Pause: mouse.locked={mouse.locked}, mouse.visible={mouse.visible}")
        if pause_menu:
            pause_menu.show()

    def resume_game(*, lock_mouse: bool = True) -> None:
        was_paused = game_state["paused"]
        game_state["paused"] = False
        if pause_menu:
            pause_menu.hide()
        if lock_mouse:
            mouse.locked = True
            mouse.visible = False
            logger.info(f"Resume: locked mouse, mouse.locked={mouse.locked}, mouse.visible={mouse.visible}")
        elif was_paused:
            # Ensure visibility for menus when resuming without locking mouse
            mouse.locked = False
            mouse.visible = True
            logger.info(f"Resume: keeping mouse unlocked, mouse.locked={mouse.locked}, mouse.visible={mouse.visible}")

    class PauseMenu:
        def __init__(self, *, on_main_menu, on_options, on_resume) -> None:
            self.on_main_menu = on_main_menu
            self.on_options = on_options
            self.on_resume = on_resume

            self.root = Entity(parent=camera.ui, enabled=False)
            self.root.z = -10

            self.overlay = Entity(
                parent=self.root,
                model="quad",
                color=color.rgba(0, 0, 0, 160),
                scale=(window.aspect_ratio * 2, 2),
                z=0,
            )

            self.panel = Entity(
                parent=self.root,
                model="quad",
                color=color.rgb(40, 40, 55),
                scale=(0.6, 0.5),
                z=-1,
            )

            self.title = Text(
                parent=self.panel,
                text="Paused",
                position=(0, 0.18),
                origin=(0, 0),
                scale=1.8,
                color=color.rgb(255, 255, 255),
                thickness=2,
            )

            button_kwargs = {
                "parent": self.panel,
                "model": "quad",
                "scale": (0.4, 0.12),
                "color": color.rgb(70, 70, 90),
                "highlight_color": color.rgb(100, 100, 130),
                "pressed_color": color.rgb(50, 50, 70),
                "origin": (0, 0),
            }

            self.message = Text(
                parent=self.panel,
                text="Options menu coming soon",
                origin=(0, 0),
                position=(0, -0.28),
                scale=1.1,
                color=color.rgb(200, 200, 200),
            )
            self.message.enabled = False

            def make_button(label: str, y: float, handler) -> Button:
                button = Button(**button_kwargs, text=label, position=(0, y))
                button.on_click = handler
                return button

            make_button(
                "Return to Game",
                0.05,
                lambda: self.on_resume(),
            )

            make_button(
                "Options",
                -0.1,
                self._handle_options,
            )

            make_button(
                "Main Menu",
                -0.25,
                lambda: self.on_main_menu(),
            )

        def _handle_options(self) -> None:
            self.message.enabled = True
            if callable(self.on_options):
                self.on_options()

        def show(self) -> None:
            self.root.enable()
            self.message.enabled = False

        def hide(self) -> None:
            self.root.disable()

    class Hotbar(Entity):  # type: ignore[misc]
        """Hotbar UI for quick access to blocks in creative mode."""
        def __init__(self, slot_count: int = 6, **kwargs):
            super().__init__(parent=camera.ui, z=-10)
            
            self.slot_count = slot_count
            self.selected_slot = 0
            self.slots = []
            # Start with an empty hotbar; blocks are added via the BlockGUI
            self.slot_items = [None for _ in range(slot_count)]
            self.slot_counts = [0 for _ in range(slot_count)]
            self.slot_icons = []  # Store 3D block previews
            self.slot_count_labels = []
            
            # Hotbar dimensions
            slot_size = 0.08
            slot_spacing = 0.09
            total_width = slot_count * slot_spacing
            
            # Hotbar background - using a placeholder quad with tinted color
            self.background = Entity(
                parent=self,
                model='quad',
                scale=(total_width + 0.02, slot_size + 0.02),
                color=color.rgba(40, 40, 40, 180),
                position=(0, -0.43, 0),
                origin=(0, 0)
            )
            
            # Create slots
            for i in range(slot_count):
                x_pos = (i - (slot_count - 1) / 2) * slot_spacing
                
                # Slot background
                slot = Entity(
                    parent=self,
                    model='quad',
                    scale=slot_size,
                    color=color.rgba(80, 80, 80, 200),
                    position=(x_pos, -0.43, -0.01),
                    origin=(0, 0)
                )
                self.slots.append(slot)
                
                # Slot number label
                Text(
                    parent=slot,
                    text=str(i + 1),
                    scale=8,
                    color=color.white,
                    origin=(0.4, -0.4),
                    z=-0.1
                )
                # Stack count label (TOP-RIGHT of the slot, inside the block area)
                # In Ursina UI, more negative z is drawn in front for siblings.
                # Put this in front of both the slot quad (z ~ -0.01) and item icon (z ~ -0.15).
                count_label = Text(
                    parent=slot,
                    text='',
                    scale=11,  # Slightly larger for bold appearance
                    color=color.white,
                    origin=(0.5, 0.5),
                    # Top-right-ish but clearly inside the slot (no clipping)
                    position=(slot_size * 0.30, slot_size * 0.18),
                    z=-0.5
                )
                # Add a black outline so the count is readable on any background
                count_label.outline_color = color.black
                count_label.outline_scale = 0.35
                self.slot_count_labels.append(count_label)
                
                # Create 3D block preview for this slot
                block_type = self.slot_items[i]
                if block_type:
                    self._create_block_icon(i, block_type)
                else:
                    self.slot_icons.append(None)
            for i in range(self.slot_count):
                self._update_slot_count_label(i)
            
            # Selection indicator
            self.selection_indicator = Entity(
                parent=self,
                model='quad',
                scale=slot_size + 0.01,
                color=color.rgba(255, 255, 255, 100),
                position=(self.slots[0].position),
                origin=(0, 0),
                z=-0.02
            )
            
            # Block name display above hotbar
            self.block_name_text = Text(
                parent=camera.ui,
                text='',
                position=(0, -0.35),
                scale=1.5,
                color=color.black,
                origin=(0, 0)
            )
            self.text_fade_time = 0  # Timer for fading text
            self.text_visible_duration = 2.5  # How long text stays visible (seconds)
            self.text_fade_duration = 0.5  # How long fade animation takes (seconds)
            # Show initial block name
            if self.slot_items[0]:
                self.block_name_text.text = self.slot_items[0].capitalize()
                self.text_fade_time = 0
        
        def _create_block_icon(self, slot_index: int, block_type: str) -> None:
            """Create a 3D block preview icon in the specified slot."""
            if slot_index >= len(self.slots):
                return
            
            slot = self.slots[slot_index]
            textures = block_type_textures.get(block_type, face_textures)
            
            # Load display settings from JSON
            display_settings = load_block_display_settings(block_type)
            
            # Default GUI display settings (Minecraft-like isometric view)
            rotation = Vec3(30, 45, 0)  # Default isometric angle
            scale_factor = 0.42  # Scale for hotbar to fill slot properly
            
            if display_settings:
                # Use rotation from JSON if available
                json_rotation = display_settings.get("rotation", [30, 45, 0])
                rotation = Vec3(json_rotation[0], json_rotation[1], json_rotation[2])
                
                # Use scale from JSON if available - scale up to fill hotbar slot
                json_scale = display_settings.get("scale", [0.57227, 0.57227, 0.57227])
                # Scale to fill hotbar slot properly
                scale_factor = 0.42
            
            # Create parent container for the 3D block preview
            icon_container = Entity(
                parent=slot,
                position=(0, 0, -0.15),
                scale=scale_factor,
                rotation=rotation
            )
            
            # Create block faces as in the main Block class but smaller
            for face_name, transform in face_transforms.items():
                texture = textures.get(face_name)
                if texture:
                    face_quad = Entity(
                        parent=icon_container,
                        model='quad',
                        texture=texture,
                        color=color.white,
                        double_sided=True,
                        position=transform['position'],
                        rotation=transform['rotation'],
                        scale=1,
                        origin=(0, 0),
                    )
                    face_quad.set_light_off()
                    
                    # Apply texture flipping
                    if face_name == 'north':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'south':
                        face_quad.texture_scale = (-1, 1)
                    elif face_name == 'east':
                        face_quad.texture_scale = (-1, 1)
                    elif face_name == 'west':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'up':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'down':
                        face_quad.texture_scale = (1, -1)
            
            # Update slot_icons list at the correct index
            if slot_index < len(self.slot_icons):
                self.slot_icons[slot_index] = icon_container
            else:
                self.slot_icons.append(icon_container)
            
        def update(self):
            """Update hotbar text fade animation."""
            # Update fade timer
            self.text_fade_time += time.dt
            
            # Calculate alpha based on fade time
            if self.text_fade_time < self.text_visible_duration:
                # Text is fully visible
                alpha = 1.0
            elif self.text_fade_time < self.text_visible_duration + self.text_fade_duration:
                # Text is fading out
                fade_progress = (self.text_fade_time - self.text_visible_duration) / self.text_fade_duration
                alpha = 1.0 - fade_progress
            else:
                # Text is fully faded
                alpha = 0.0
            
            # Apply alpha to text color (RGBA values 0-255)
            self.block_name_text.color = color.rgba(0, 0, 0, int(alpha * 255))
        
        def select_slot(self, slot_index: int) -> None:
            """Select a hotbar slot by index (0-based)."""
            if 0 <= slot_index < self.slot_count:
                self.selected_slot = slot_index
                self.selection_indicator.position = self.slots[slot_index].position
                # Update block name display above hotbar
                block_type = self.slot_items[slot_index]
                if block_type:
                    self.block_name_text.text = block_type.capitalize()
                    # Reset fade timer to show text again
                    self.text_fade_time = 0
                    self.block_name_text.color = color.black
                else:
                    self.block_name_text.text = ''
                logger.info(f"Selected hotbar slot {slot_index + 1}")
        
        def _update_slot_count_label(self, slot_index: int) -> None:
            if hasattr(self, "slot_count_labels") and 0 <= slot_index < len(self.slot_count_labels):
                label = self.slot_count_labels[slot_index]
                count = 0
                if hasattr(self, "slot_counts") and slot_index < len(self.slot_counts):
                    count = self.slot_counts[slot_index]
                # Only show a number when there is more than 1 item in the stack
                label.text = str(count) if count > 1 else ''
                
        def get_selected_item(self) -> Optional[str]:
            """Get the block type in the currently selected slot."""
            return self.slot_items[self.selected_slot]
            
        def set_slot_item(self, slot_index: int, item_type: Optional[str]) -> None:
            """Set the item in a specific slot and update the visual."""
            if 0 <= slot_index < self.slot_count:
                self.slot_items[slot_index] = item_type
                if hasattr(self, "slot_counts") and slot_index < len(self.slot_counts):
                    if item_type is None:
                        self.slot_counts[slot_index] = 0
                    elif self.slot_counts[slot_index] <= 0:
                        self.slot_counts[slot_index] = 1
                
                # Destroy old icon if exists
                if slot_index < len(self.slot_icons) and self.slot_icons[slot_index]:
                    destroy(self.slot_icons[slot_index])
                    self.slot_icons[slot_index] = None
                
                # Create new icon if item_type is not None
                if item_type:
                    self._create_block_icon(slot_index, item_type)
                if hasattr(self, "_update_slot_count_label"):
                    self._update_slot_count_label(slot_index)

        def add_item(self, block_type: str, max_stack: int = 70) -> int:
            if not hasattr(self, "slot_counts"):
                for slot_index in range(self.slot_count):
                    if self.slot_items[slot_index] is None:
                        self.set_slot_item(slot_index, block_type)
                        return slot_index
                return -1
            for slot_index in range(self.slot_count):
                if self.slot_items[slot_index] == block_type and self.slot_counts[slot_index] < max_stack:
                    self.slot_counts[slot_index] += 1
                    if hasattr(self, "_update_slot_count_label"):
                        self._update_slot_count_label(slot_index)
                    return slot_index
            for slot_index in range(self.slot_count):
                if self.slot_items[slot_index] is None:
                    self.set_slot_item(slot_index, block_type)
                    if slot_index < len(self.slot_counts):
                        self.slot_counts[slot_index] = 1
                        if hasattr(self, "_update_slot_count_label"):
                            self._update_slot_count_label(slot_index)
                    return slot_index
            return -1

    class BlockGUI(Entity):  # type: ignore[misc]
        """Block selection GUI for choosing blocks to place in hotbar."""
        def __init__(self, hotbar_ref, block_textures_dict, player_ref=None, **kwargs):
            super().__init__(parent=camera.ui)
            self.hotbar_ref = hotbar_ref
            self.block_textures = block_textures_dict
            self.player_ref = player_ref  # Reference to player controller
            self.visible = False
            self.widgets = []
            self.block_slots = []
            self.search_query = ""
            
            # Drag and drop state
            self.dragging = False
            self.dragged_block_type = None
            self.drag_entity = None
            self.drag_stack_count = 0  # How many items are currently on the mouse
            self.drag_count_label = None  # Text label showing stack count on the drag icon
            
            # Guard flags to prevent recursive calls
            self._hiding = False
            self._showing = False
            
            # Initialize stored sensitivity to default
            self._stored_mouse_sensitivity = Vec2(40, 40)
            
            # Available blocks (can be expanded in the future)
            self.all_blocks = ['grass', 'dirt']  # Will expand this list
            self.filtered_blocks = self.all_blocks.copy()
            
            # Background panel - load texture explicitly
            try:
                from ursina import Texture
                bg_texture = Texture('assets/menu/GUIS/Gui_grey.png')
                self.background = Entity(
                    parent=self,
                    model='quad',
                    texture=bg_texture,
                    scale=(1.2, 0.9),
                    position=(0, 0.05),  # Moved up to show hotbar
                    z=-1,
                    color=color.white
                )
            except:
                # Fallback to colored background if texture fails
                logger.warning("Failed to load Gui_grey.png, using solid color")
                self.background = Entity(
                    parent=self,
                    model='quad',
                    color=color.rgb(140, 140, 150),
                    scale=(1.2, 0.9),
                    position=(0, 0.05),  # Moved up to show hotbar
                    z=-1
                )
            self.widgets.append(self.background)
            
            # Title
            self.title = Text(
                parent=self,
                text='Block Selection',
                position=(0, 0.43),  # Moved up
                origin=(0, 0),
                scale=1.8,
                color=color.black,
                z=-2
            )
            self.widgets.append(self.title)
            
            # Search bar background
            self.search_bg = Entity(
                parent=self,
                model='quad',
                color=color.rgb(60, 60, 70),
                scale=(0.8, 0.08),
                position=(0, 0.33),  # Moved up
                z=-2
            )
            self.widgets.append(self.search_bg)
            
            # Search text label
            self.search_label = Text(
                parent=self,
                text='Search:',
                position=(-0.42, 0.33),  # Moved up
                origin=(0, 0),
                scale=1.2,
                color=color.black,
                z=-3
            )
            self.widgets.append(self.search_label)
            
            # Search input field
            self.search_field = InputField(
                parent=self,
                position=(0.05, 0.33),  # Moved up
                z=-3,
                max_lines=1
            )
            self.search_field.scale = (0.6, 0.06)
            self.search_field.on_value_changed = self.on_search_changed
            self.widgets.append(self.search_field)
            
            # Close button
            self.close_text = Text(
                parent=self,
                text='[Press E or ESC to close]',
                position=(0, -0.37),  # Moved up
                origin=(0, 0),
                scale=1.0,
                color=color.gray,
                z=-2
            )
            self.widgets.append(self.close_text)
            
            # Create block grid
            self.create_block_grid()
            
            # Start hidden (visible and enabled already default to False)
            # Don't call hide() here as it can cause issues during initialization
            self.enabled = False
        
        def create_block_grid(self):
            """Create a grid of block selection slots."""
            # Clear existing slots
            for slot in self.block_slots:
                destroy(slot['bg'])
                destroy(slot['icon'])
                destroy(slot['button'])
                if 'text' in slot:
                    destroy(slot['text'])
            self.block_slots = []
            
            # Grid parameters
            cols = 6
            rows = 3
            slot_size = 0.12
            slot_spacing = 0.14
            start_x = -(cols - 1) * slot_spacing / 2
            start_y = 0.17  # Moved up to match repositioned GUI
            
            # Create slots for filtered blocks
            for i, block_type in enumerate(self.filtered_blocks):
                if i >= cols * rows:
                    break  # Limit to grid size for now
                
                row = i // cols
                col = i % cols
                x_pos = start_x + col * slot_spacing
                y_pos = start_y - row * slot_spacing
                
                # Slot background - match hotbar style
                slot_bg = Entity(
                    parent=self,
                    model='quad',
                    color=color.rgba(80, 80, 80, 200),  # Match hotbar color
                    scale=(slot_size, slot_size),
                    position=(x_pos, y_pos),
                    z=-2  # Behind icons (less negative = further back in UI)
                )
                
                # Create 3D block icon using face quads (like hotbar)
                icon_container = self._create_3d_block_icon(block_type, x_pos, y_pos)
                logger.info(f"Created icon for {block_type} at ({x_pos}, {y_pos})")
                
                # Block name text
                block_text = Text(
                    parent=self,
                    text=block_type.capitalize(),
                    position=(x_pos, y_pos - slot_size * 0.6),
                    origin=(0, 0),
                    scale=0.8,
                    color=color.black,
                    z=-6  # In front of icons (more negative = closer)
                )
                
                # Interactive button - in front of everything to capture clicks
                # Using drag-and-drop only (no on_click to avoid conflicts)
                slot_button = Button(
                    parent=self,
                    model='quad',
                    color=color.rgba(0, 0, 0, 0),  # Transparent
                    scale=(slot_size, slot_size),
                    position=(x_pos, y_pos),
                    z=-7  # Most negative = in front to capture hover/drag
                )
                # Store button reference for hover effects
                slot_button.block_type = block_type
                slot_button.on_mouse_enter = lambda btn=slot_button: self.on_slot_hover(btn)
                slot_button.on_mouse_exit = lambda btn=slot_button: self.on_slot_exit(btn)
                
                self.block_slots.append({
                    'bg': slot_bg,
                    'icon': icon_container,
                    'text': block_text,
                    'button': slot_button,
                    'block_type': block_type
                })
        
        def on_search_changed(self):
            """Handle search input changes."""
            search_text = self.search_field.text.lower()
            self.filtered_blocks = [b for b in self.all_blocks if search_text in b.lower()]
            self.create_block_grid()
        
        def _create_3d_block_icon(self, block_type, x_pos, y_pos):
            """Create a 3D block preview using face quads like the hotbar."""
            # Face transforms for cube faces
            face_transforms = {
                'north': {'position': (0, 0, 0.5), 'rotation': (0, 0, 0)},
                'south': {'position': (0, 0, -0.5), 'rotation': (0, 180, 0)},
                'east': {'position': (0.5, 0, 0), 'rotation': (0, 90, 0)},
                'west': {'position': (-0.5, 0, 0), 'rotation': (0, -90, 0)},
                'up': {'position': (0, 0.5, 0), 'rotation': (90, 0, 0)},
                'down': {'position': (0, -0.5, 0), 'rotation': (-90, 0, 0)},
            }
            
            # Get textures directly from block_textures dict
            textures = self.block_textures.get(block_type, {})
            
            if not textures:
                logger.warning(f"No textures found for block type: {block_type}")
            
            # Default rotation for GUI display
            rotation = Vec3(30, 45, 0)
            scale_factor = 0.04  # Scale for inventory grid icons
            
            # Create parent container
            icon_container = Entity(
                parent=self,
                position=(x_pos, y_pos),  # x, y only
                z=-4,  # Set z separately to match other UI elements
                scale=scale_factor,
                rotation=rotation,
                enabled=True,
                visible=True
            )
            
            # Create block faces
            faces_created = 0
            for face_name, transform in face_transforms.items():
                texture = textures.get(face_name)
                if texture:
                    face_quad = Entity(
                        parent=icon_container,
                        model='quad',
                        texture=texture,
                        color=color.white,
                        double_sided=True,
                        position=transform['position'],
                        rotation=transform['rotation'],
                        scale=1,
                        origin=(0, 0),
                        enabled=True,
                        visible=True
                    )
                    face_quad.set_light_off()
                    
                    # Apply texture flipping (same as hotbar)
                    if face_name == 'north':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'south':
                        face_quad.texture_scale = (-1, 1)
                    elif face_name == 'east':
                        face_quad.texture_scale = (-1, 1)
                    elif face_name == 'west':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'up':
                        face_quad.texture_scale = (1, 1)
                    elif face_name == 'down':
                        face_quad.texture_scale = (1, -1)
                    faces_created += 1
                else:
                    # Create colored fallback if texture is missing
                    fallback_color = color.rgb(100, 200, 100) if block_type == 'grass' else color.rgb(139, 69, 19)
                    face_quad = Entity(
                        parent=icon_container,
                        model='quad',
                        color=fallback_color,
                        double_sided=True,
                        position=transform['position'],
                        rotation=transform['rotation'],
                        scale=1,
                        origin=(0, 0),
                        enabled=True,
                        visible=True
                    )
                    faces_created += 1
            
            if faces_created == 0:
                # Create a simple colored cube as absolute fallback
                logger.warning(f"No faces created for {block_type}, creating fallback cube")
                fallback_color = color.rgb(100, 200, 100) if block_type == 'grass' else color.rgb(139, 69, 19)
                Entity(
                    parent=icon_container,
                    model='cube',
                    color=fallback_color,
                    scale=1,
                    enabled=True,
                    visible=True
                )
            
            logger.info(f"Created icon for {block_type} with {faces_created} faces at z={icon_container.z}")
            return icon_container
        
        def start_drag(self, block_type):
            """Start dragging a block."""
            self.dragging = True
            self.dragged_block_type = block_type
            self.drag_stack_count = 1  # Start with a single item on the mouse
            
            # Create drag entity that follows mouse (larger for visibility)
            self.drag_entity = self._create_3d_block_icon(block_type, mouse.x, mouse.y)
            self.drag_entity.z = -10  # Render in front of most UI elements
            # Simple grow animation when the item is picked up
            original_scale = self.drag_entity.scale
            self.drag_entity.scale = original_scale * 0.7
            try:
                self.drag_entity.animate_scale(original_scale * 1.1, duration=0.12)
            except Exception:
                # Fallback in case animate_scale is unavailable
                self.drag_entity.scale = original_scale * 1.1
            self.drag_entity.enabled = True
            self.drag_entity.visible = True
            # Enable all face quad children
            for child in self.drag_entity.children:
                child.enabled = True
                child.visible = True

            # Create or update the drag stack count label so it rides on the dragged icon
            # Use similar style to the hotbar stack label, but parented to the drag entity
            if self.drag_count_label is None:
                from ursina import Text  # Local import to avoid top-level changes
                self.drag_count_label = Text(
                    parent=self.drag_entity,
                    text='',
                    scale=24,              # Bigger so the number is clearly visible
                    color=color.white,
                    origin=(0.5, 0.5),
                    position=(0.25, 0.25),  # Top-right-ish on the dragged icon
                    z=-0.5,                 # Slightly in front of the icon faces
                )
                # Add a black outline for readability
                self.drag_count_label.outline_color = color.black
                self.drag_count_label.outline_scale = 0.4
            else:
                # If reusing an existing label, re-parent it to the new drag_entity
                self.drag_count_label.parent = self.drag_entity
                self.drag_count_label.position = (0.25, 0.25)
                self.drag_count_label.z = -0.5

            # Always show the current mouse stack count (helps visibility while dragging)
            self.drag_count_label.text = str(self.drag_stack_count)
            logger.info(f"Started dragging: {block_type} at mouse position ({mouse.x}, {mouse.y})")
        
        def on_slot_hover(self, slot):
            """Handle mouse hover over slot."""
            if not self.dragging:
                slot.color = color.rgba(255, 255, 255, 50)  # Slight highlight
        
        def on_slot_exit(self, slot):
            """Handle mouse exit from slot."""
            if not self.dragging:
                slot.color = color.rgba(0, 0, 0, 0)  # Transparent
        
        def update(self):
            """Update drag entity position and highlight drop targets."""
            if self.dragging and self.drag_entity:
                # Make drag entity follow mouse
                self.drag_entity.position = (mouse.x, mouse.y, -10)
                # drag_count_label is parented to drag_entity, so it will follow automatically
                
                # Highlight hotbar slot under mouse
                for i, slot in enumerate(self.hotbar_ref.slots):
                    slot_x = slot.x
                    slot_y = slot.y
                    slot_half_size = slot.scale_x / 2 if hasattr(slot, 'scale_x') else 0.04
                    
                    # Check if mouse is over this slot
                    if (abs(mouse.x - slot_x) < slot_half_size and 
                        abs(mouse.y - slot_y) < slot_half_size):
                        # Highlight this slot
                        slot.color = color.rgba(150, 150, 255, 150)
                    else:
                        # Reset to normal color
                        slot.color = color.rgba(80, 80, 80, 200)
        
        def _drop_dragged_to_hotbar(self, single: bool) -> None:
            """Drop the currently dragged block onto the hotbar, if over a slot.
            single=True places 1 item; single=False places a full stack (70).
            """
            if not self.dragging or not self.drag_entity:
                return

            hotbar = self.hotbar_ref
            dropped = False
            # How many items to place in the slot
            desired_count = min(self.drag_stack_count, 70) if single else 70

            for i, slot in enumerate(hotbar.slots):
                # Check if mouse is within slot bounds
                slot_x = slot.x
                slot_y = slot.y
                slot_half_size = slot.scale_x / 2 if hasattr(slot, 'scale_x') else 0.045
                if (abs(mouse.x - slot_x) < slot_half_size and 
                        abs(mouse.y - slot_y) < slot_half_size):
                    # We are over this hotbar slot
                    if hasattr(hotbar, "slot_counts") and i < len(hotbar.slot_counts):
                        # Ensure the slot has the dragged block type
                        current_item = hotbar.slot_items[i]
                        if current_item is None or current_item != self.dragged_block_type:
                            hotbar.set_slot_item(i, self.dragged_block_type)

                        # Set the stack size in the slot to the desired amount
                        hotbar.slot_counts[i] = desired_count

                        if hasattr(hotbar, "_update_slot_count_label"):
                            hotbar._update_slot_count_label(i)
                    else:
                        hotbar.set_slot_item(i, self.dragged_block_type)

                    hotbar.select_slot(i)
                    if self.player_ref is not None and hasattr(self.player_ref, "selected_block_type"):
                        self.player_ref.selected_block_type = self.dragged_block_type
                    logger.info(
                        f"Drag-drop {'single' if single else 'stack'} {self.dragged_block_type} "
                        f"into hotbar slot {i + 1}"
                    )
                    dropped = True
                    break

            # Clean up drag entity and reset highlights regardless of whether we dropped successfully
            if self.drag_entity:
                destroy(self.drag_entity)
                self.drag_entity = None
            # Destroy and reset drag count label so a fresh one is created next time
            if self.drag_count_label is not None:
                destroy(self.drag_count_label)
                self.drag_count_label = None

            for slot in hotbar.slots:
                slot.color = color.rgba(80, 80, 80, 200)

            self.dragging = False
            self.dragged_block_type = None
        
        def input(self, key):
            """Handle input for drag and drop."""
            # Left click behavior:
            # - If not dragging: single-click on a block in the grid to pick it up (sticks to cursor with 1 item).
            # - If dragging: click on a hotbar slot to place items equal to drag_stack_count there.
            #                clicking another block in the grid while dragging will ADD one more of that block
            #                to the mouse stack (up to 70) and update the cursor count text.
            if key == 'left mouse down':
                if not self.dragging:
                    # Start drag from any hovered block slot
                    for slot in self.block_slots:
                        btn = slot['button']
                        if btn.hovered:
                            self.start_drag(slot['block_type'])
                            logger.info(f"Starting drag for {slot['block_type']}")
                            break
                else:
                    # We are already dragging.
                    # First, see if we're clicking another block in the grid to increase the mouse stack.
                    hovered_block = None
                    for slot in self.block_slots:
                        btn = slot['button']
                        if btn.hovered:
                            hovered_block = slot['block_type']
                            break

                    if hovered_block is not None and hovered_block == self.dragged_block_type:
                        # Add one more of this block to the mouse stack, up to max 70
                        if self.drag_stack_count < 70:
                            self.drag_stack_count += 1
                        # Update drag count label (hide when == 1 for consistency with hotbar)
                        if self.drag_count_label is not None:
                            self.drag_count_label.text = (
                                str(self.drag_stack_count) if self.drag_stack_count > 1 else ''
                            )
                        logger.info(
                            f"Incremented drag stack for {self.dragged_block_type} to {self.drag_stack_count}"
                        )
                    else:
                        # Not clicking another grid block of the same type, so drop onto the hotbar
                        self._drop_dragged_to_hotbar(single=True)

            # Right-click behavior:
            # - If dragging and hovering a block in the grid with the same type, set the mouse stack to max (70).
            # - Else if dragging, drop a FULL STACK into the hotbar slot under the cursor.
            # - If not dragging: preserve existing behavior (full stack into selected slot in Creative).
            if key == 'right mouse down':
                if self.dragging:
                    # Check if we're on another block in the grid of the same type;
                    # if so, treat this as "fill mouse stack to max".
                    hovered_block = None
                    for slot in self.block_slots:
                        btn = slot['button']
                        if btn.hovered:
                            hovered_block = slot['block_type']
                            break

                    if hovered_block is not None and hovered_block == self.dragged_block_type:
                        self.drag_stack_count = 70
                        if self.drag_count_label is not None:
                            self.drag_count_label.text = str(self.drag_stack_count)
                        logger.info(
                            f"Set drag stack for {self.dragged_block_type} to max (70) via right-click"
                        )
                    else:
                        # Not hovering a grid block of the same type; drop full stack to hotbar
                        self._drop_dragged_to_hotbar(single=False)
                    return

                # Only allowed when the player is a CreativeController
                if self.player_ref is not None and getattr(self.player_ref.__class__, '__name__', '') == 'CreativeController':
                    for slot in self.block_slots:
                        btn = slot['button']
                        if btn.hovered:
                            block_type = slot['block_type']
                            hotbar = self.hotbar_ref
                            target_slot = hotbar.selected_slot
                            hotbar.set_slot_item(target_slot, block_type)
                            if hasattr(hotbar, "slot_counts") and target_slot < len(hotbar.slot_counts):
                                hotbar.slot_counts[target_slot] = 70
                                if hasattr(hotbar, "_update_slot_count_label"):
                                    hotbar._update_slot_count_label(target_slot)
                            hotbar.select_slot(target_slot)
                            if hasattr(self.player_ref, "selected_block_type"):
                                self.player_ref.selected_block_type = block_type
                            logger.info(f"Right-click full stack: set {block_type} x70 in hotbar slot {target_slot + 1}")
                            break

            # Number keys 1-6 while hovering a block: assign that block to the corresponding hotbar slot
            if key in ('1', '2', '3', '4', '5', '6'):
                # Only meaningful if we're hovering a block slot
                hovered_block_type = None
                for slot in self.block_slots:
                    btn = slot['button']
                    if btn.hovered:
                        hovered_block_type = slot['block_type']
                        break
                if hovered_block_type is None:
                    return

                hotbar = self.hotbar_ref
                slot_index = int(key) - 1
                if 0 <= slot_index < hotbar.slot_count:
                    # Set the item in that hotbar slot
                    hotbar.set_slot_item(slot_index, hovered_block_type)
                    if hasattr(hotbar, "slot_counts") and slot_index < len(hotbar.slot_counts):
                        # Using the number keys should only place a single block in that slot
                        hotbar.slot_counts[slot_index] = 1
                        if hasattr(hotbar, "_update_slot_count_label"):
                            hotbar._update_slot_count_label(slot_index)
                    hotbar.select_slot(slot_index)
                    if self.player_ref is not None and hasattr(self.player_ref, "selected_block_type"):
                        self.player_ref.selected_block_type = hovered_block_type
                    logger.info(f"Hotkey {key}: set {hovered_block_type} in hotbar slot {slot_index + 1}")
        
        def select_block(self, block_type):
            """Add selected block to the first empty hotbar slot or selected slot."""
            logger.info(f"Selected block: {block_type}")
            
            # Try to add to current hotbar slot
            current_slot = self.hotbar_ref.selected_slot
            self.hotbar_ref.set_slot_item(current_slot, block_type)
            if hasattr(self.hotbar_ref, "slot_counts") and current_slot < len(self.hotbar_ref.slot_counts):
                self.hotbar_ref.slot_counts[current_slot] = 70
                if hasattr(self.hotbar_ref, "_update_slot_count_label"):
                    self.hotbar_ref._update_slot_count_label(current_slot)
            self.hotbar_ref.select_slot(current_slot)  # Refresh display
            
            logger.info(f"Added {block_type} to hotbar slot {current_slot + 1}")
        
        def show(self):
            """Show the block GUI."""
            # Guard against recursive calls
            if self._showing:
                return
            self._showing = True
            
            self.visible = True
            self.enabled = True
            for widget in self.widgets:
                if hasattr(widget, 'enable'):
                    widget.enable()
                else:
                    widget.enabled = True
            
            # Enable all slot elements including icon children
            for slot in self.block_slots:
                slot['bg'].enabled = True
                slot['text'].enabled = True
                slot['button'].enabled = True
                
                # Enable icon and all its children (face quads)
                icon = slot['icon']
                if icon:
                    icon.enabled = True
                    icon.visible = True
                    # Enable all child entities (the face quads)
                    for child in icon.children:
                        child.enabled = True
                        child.visible = True
            
            mouse.locked = False
            mouse.visible = True
            
            # Disable player controller movement and camera rotation
            if self.player_ref and hasattr(self.player_ref, 'mouse_sensitivity'):
                current_sensitivity = self.player_ref.mouse_sensitivity
                # Only store if it's a valid (non-zero) sensitivity
                if current_sensitivity and (current_sensitivity.x != 0 or current_sensitivity.y != 0):
                    self._stored_mouse_sensitivity = current_sensitivity
                    logger.info(f"Stored mouse sensitivity: {self._stored_mouse_sensitivity}")
                elif not hasattr(self, '_stored_mouse_sensitivity'):
                    # Set default if we don't have one stored already
                    self._stored_mouse_sensitivity = Vec2(40, 40)
                    logger.info(f"No valid sensitivity to store, using default: {self._stored_mouse_sensitivity}")
                self.player_ref.mouse_sensitivity = Vec2(0, 0)
                logger.info(f"Disabled mouse sensitivity -> (0, 0)")
            
            logger.info(f"Block GUI opened with {len(self.block_slots)} block slots")
            
            # Reset guard flag
            self._showing = False
        
        def hide(self):
            """Hide the block GUI."""
            # Guard against recursive calls
            if self._hiding:
                return
            self._hiding = True
            
            self.visible = False
            self.enabled = False
            for widget in self.widgets:
                if hasattr(widget, 'disable'):
                    widget.disable()
                else:
                    widget.enabled = False
            for slot in self.block_slots:
                slot['bg'].enabled = False
                slot['icon'].enabled = False
                slot['text'].enabled = False
                slot['button'].enabled = False
            
            # Restore mouse state - lock and hide cursor for gameplay
            mouse.locked = True
            mouse.visible = False
            
            # Restore player controller mouse sensitivity
            if self.player_ref and hasattr(self.player_ref, 'mouse_sensitivity'):
                if hasattr(self, '_stored_mouse_sensitivity'):
                    restore_sensitivity = self._stored_mouse_sensitivity
                    # Double-check the stored value is valid (non-zero)
                    if not restore_sensitivity or (restore_sensitivity.x == 0 and restore_sensitivity.y == 0):
                        restore_sensitivity = Vec2(40, 40)
                        logger.warning("Stored sensitivity was invalid (0, 0), using default (40, 40)")
                    self.player_ref.mouse_sensitivity = restore_sensitivity
                    logger.info(f"Restored mouse sensitivity: {restore_sensitivity}")
                else:
                    # Fallback if sensitivity wasn't stored
                    self.player_ref.mouse_sensitivity = Vec2(40, 40)
                    logger.warning("No stored mouse sensitivity, using default (40, 40)")
            
            logger.info("Block GUI closed")
            
            # Reset guard flag
            self._hiding = False

    # Track all dropped items for pickup
    dropped_items = []
    
    class DroppedItem(Entity):  # type: ignore[misc]
        """Dropped item entity that rotates, bobs, and eventually despawns."""
        def __init__(self, block_type: str, position: Vec3, **kwargs):
            super().__init__(model=None, collider=None, position=position)
            self.block_type = block_type
            
            # Visual properties
            self.rotation_speed = 57.6  # Degrees per second (Minecraft-like)
            self.bob_time = 0
            self.bob_amplitude = 0.1  # How high/low the item bobs
            self.bob_frequency = 2.0  # Bobs per second
            self.initial_y = position.y
            
            # Simple physics for thrown/falling motion
            self.velocity = Vec3(0, 0, 0)
            self.gravity_strength = 18.0
            self.is_falling = True

            # Give the item a small forward/upward impulse so it arcs out from the player
            forward = Vec3(camera.forward.x, 0, camera.forward.z)
            if forward.length() > 0:
                forward = forward.normalized()
            throw_speed = 4.0
            self.velocity = forward * throw_speed + Vec3(0, 3.0, 0)
            
            # Despawn timer (5 minutes = 300 seconds)
            self.despawn_timer = 300.0
            self.can_be_picked_up = False
            self.pickup_delay = 1.3  # Slightly longer delay before pickup (seconds)
            
            # Create 3D block model at 1/4 scale (Minecraft-like)
            self.model_container = Entity(
                parent=self,
                position=(0, 0, 0),
                scale=0.25  # 1/4 scale
            )
            
            # Load block display settings from JSON
            display_settings = load_block_display_settings(block_type)
            if display_settings:
                rotation = display_settings.get('rotation', [0, 0, 0])
                self.model_container.rotation = Vec3(rotation[0], rotation[1], rotation[2])
            
            # Get textures for this block type
            textures = block_type_textures.get(block_type, face_textures)
            
            # Create 6 faces for the 3D block
            face_data = [
                ('north', Vec3(0, 0, -0.5), (0, 0, 0)),
                ('south', Vec3(0, 0, 0.5), (0, 180, 0)),
                ('east', Vec3(0.5, 0, 0), (0, 90, 0)),
                ('west', Vec3(-0.5, 0, 0), (0, -90, 0)),
                ('up', Vec3(0, 0.5, 0), (90, 0, 0)),
                ('down', Vec3(0, -0.5, 0), (-90, 0, 0))
            ]
            
            for face_name, face_pos, face_rot in face_data:
                texture = textures.get(face_name)
                if texture:
                    face = Entity(
                        parent=self.model_container,
                        model='quad',
                        texture=texture,
                        position=face_pos,
                        rotation=face_rot,
                        scale=1.0,
                        color=color.white,
                        double_sided=True  # Make faces visible from both sides
                    )
            
            # Add to global dropped items list
            dropped_items.append(self)
            
            logger.info(f"Dropped {block_type} item at {position}")
        
        def update(self):
            """Update rotation, bobbing, and despawn timer."""
            if game_state.get("paused", False):
                return
            
            # Rotate continuously (spin a bit faster while in the air)
            spin_multiplier = 2.0 if self.is_falling else 1.0
            self.model_container.rotation_y += self.rotation_speed * spin_multiplier * time.dt
            
            if self.is_falling:
                # Apply simple gravity-based motion while the item is in the air
                self.velocity.y -= self.gravity_strength * time.dt
                self.position += self.velocity * time.dt
                
                # Check for ground beneath and snap to it when landing
                ground_hit = raycast(
                    origin=self.world_position,
                    direction=Vec3(0, -1, 0),
                    distance=0.6,
                    ignore=(self,),
                    traverse_target=scene,
                )
                if ground_hit.hit and self.velocity.y <= 0:
                    self.is_falling = False
                    self.velocity = Vec3(0, 0, 0)
                    # Rest slightly above the hit point so it appears on top of the block
                    self.initial_y = ground_hit.world_point.y + 0.125
                    self.y = self.initial_y
                    self.bob_time = 0
            else:
                # Continuously check the ground below so the item reacts to world changes
                ground_hit = raycast(
                    origin=self.world_position,
                    direction=Vec3(0, -1, 0),
                    distance=1.0,
                    ignore=(self,),
                    traverse_target=scene,
                )
                if not ground_hit.hit:
                    # No support directly below - start falling again
                    self.is_falling = True
                else:
                    target_y = ground_hit.world_point.y + 0.125
                    # If the ground moved down significantly, let physics handle the fall
                    if target_y < self.initial_y - 0.01:
                        self.is_falling = True
                    else:
                        # Ground stayed the same height or moved up (e.g. block placed under item)
                        self.initial_y = target_y
                        # Bob up and down while resting on the ground
                        self.bob_time += time.dt * self.bob_frequency
                        bob_offset = math.sin(self.bob_time * math.pi * 2) * self.bob_amplitude
                        self.y = self.initial_y + bob_offset
            
            # Update pickup delay
            if not self.can_be_picked_up:
                self.pickup_delay -= time.dt
                if self.pickup_delay <= 0:
                    self.can_be_picked_up = True
            
            # Update despawn timer
            self.despawn_timer -= time.dt
            if self.despawn_timer <= 0:
                logger.info(f"Despawning {self.block_type} item")
                if self in dropped_items:
                    dropped_items.remove(self)
                destroy(self)
                return
            
            # Flash when about to despawn (last 10 seconds)
            if self.despawn_timer < 10:
                flash_speed = 5  # Flashes per second
                if int(self.despawn_timer * flash_speed) % 2 == 0:
                    self.model_container.visible = True
                else:
                    self.model_container.visible = False
        
        def try_pickup(self, player_pos: Vec3, pickup_range: float = 1.5) -> bool:
            """Check if player is close enough to pick up the item."""
            if not self.can_be_picked_up:
                return False
            
            offset = self.world_position - player_pos
            offset.y = 0
            distance = offset.length()
            return distance <= pickup_range
        
        def pickup(self):
            """Remove this item from the world."""
            if self in dropped_items:
                dropped_items.remove(self)
            destroy(self)

    def is_face_visible(position: Vec3, face_name: str) -> bool:
        """Check if a block face should be rendered (not hidden by adjacent block)."""
        # During world generation, render all faces
        if not enable_face_culling:
            return True
            
        # Get the offset for this face direction
        face_offsets = {
            'north': Vec3(0, 0, -1),
            'south': Vec3(0, 0, 1),
            'east': Vec3(1, 0, 0),
            'west': Vec3(-1, 0, 0),
            'up': Vec3(0, 1, 0),
            'down': Vec3(0, -1, 0),
        }
        
        # Calculate adjacent position
        adjacent_pos = position + face_offsets[face_name]
        adjacent_key = pos_to_key(adjacent_pos)
        
        # Check if there's a block at the adjacent position
        return adjacent_key not in current_world_blocks

    class Block(Entity):  # type: ignore[misc]
        def __init__(
            self,
            *,
            position: Vec3,
            block_size: float,
            textures: Dict[str, object],
            block_type: str = "grass",
            record_state: bool = True,
            mark_dirty: bool = True,
        ):
            # Create parent entity with NO model (so nothing to render) but keep it "visible"
            # This way child entities won't inherit visible=False
            super().__init__(
                position=position,
                model=None,  # No visual model - just a container
                collider='box',  # Keep collision
                scale=block_size,
            )

            # Store properties
            self.block_size = block_size
            self.textures = textures
            self.block_type = block_type
            self.grid_key = pos_to_key(position)
            self.face_entities = {}  # Store face entities for dynamic culling

            # Create each face as a separate visible quad with its own texture
            # Only create faces that are visible (not hidden by adjacent blocks)
            for face_name, transform in face_transforms.items():
                # Performance: Only render visible faces
                if not is_face_visible(position, face_name):
                    continue
                    
                texture = textures.get(face_name)
                if texture:
                    # Create quad as child entity with proper texture orientation
                    face_quad = Entity(
                        parent=self,  # Parent directly - entity is visible so children inherit visibility
                        model='quad',
                        texture=texture,
                        color=color.white,
                        double_sided=True,  # Keep double-sided for correct rendering
                        position=transform['position'],  # Local position relative to parent
                        rotation=transform['rotation'],
                        scale=1,  # Scale of 1 since parent is already scaled
                        origin=(0, 0),
                    )
                    # Configure rendering for performance
                    face_quad.set_light_off()  # Disable lighting calculations
                    
                    # Optimize texture filtering
                    if hasattr(face_quad.texture, 'setMinfilter'):
                        from panda3d.core import SamplerState
                        face_quad.texture.setMinfilter(SamplerState.FT_linear_mipmap_linear)
                        face_quad.texture.setMagfilter(SamplerState.FT_linear)
                    
                    # Apply texture flipping based on face orientation to ensure upright text
                    if face_name == 'north':
                        face_quad.texture_scale = (1, 1)   # Normal for north
                    elif face_name == 'south':
                        face_quad.texture_scale = (-1, 1)  # Flip horizontally for south
                    elif face_name == 'east':
                        face_quad.texture_scale = (-1, 1)  # Flip horizontally for east
                    elif face_name == 'west':
                        face_quad.texture_scale = (1, 1)   # Normal for west
                    elif face_name == 'up':
                        face_quad.texture_scale = (1, 1)   # Normal for top
                    elif face_name == 'down':
                        face_quad.texture_scale = (1, -1)  # Flip vertically for bottom
                    
                    # Store reference for potential dynamic culling
                    self.face_entities[face_name] = face_quad

            # Track this block
            world_entities.append(self)
            if record_state:
                add_block_to_state(self, mark_dirty=mark_dirty)

    class CreativeController(FirstPersonController):  # type: ignore[misc]
        def __init__(
            self,
            *,
            flight_speed: float = 7.0,
            height: float = 2.0,  # Exactly 2 blocks tall
            collider_height: float | None = None,
            head_clearance: float = 0.5,
            **kwargs,
        ):
            kwargs.setdefault("height", height)
            kwargs.setdefault("jump_height", 2.5)  # Jump height when not flying
            super().__init__(**kwargs)
            self.flight_speed = flight_speed
            self.gravity = 0  # Start with no gravity (flying enabled)
            self.speed = 5  # Walking speed
            # Ensure collision is enabled even with no gravity
            self.collider = 'box'
            # Enable collision detection for all movement
            self.traverse_target = scene
            # Make sure camera pivot matches custom height
            self.head_clearance = head_clearance
            self.eye_height = max(self.height - self.head_clearance, 0.5)
            if hasattr(self, "camera_pivot"):
                self.camera_pivot.y = self.eye_height
            # Adjust collider to avoid clipping; shrink slightly shorter than height
            collider_y = (collider_height or self.height) * 0.9
            self.collider = BoxCollider(self, size=Vec3(0.6, collider_y, 0.6), center=Vec3(0, collider_y / 2, 0))
            
            # Create player body model
            self.body_parts = create_player_body(self)
            logger.info("Created player model with %d body parts", len(self.body_parts))
            
            # Camera mode (first-person vs third-person)
            self.camera_mode = 'first_person'  # 'first_person' or 'third_person'
            self.third_person_distance = 5.0  # Distance behind player in third-person
            self._update_camera_mode()  # Initialize camera mode
            
            # Walking animation
            self.animation_time = 0
            self.last_position = Vec3(0, 0, 0)
            self.is_moving = False
            
            # Block interaction settings
            self.block_reach = 5.0  # How far the player can reach to place/break blocks
            self.target_block = None  # Currently targeted block
            self.target_position = None  # Position for new block placement
            self.target_normal = None  # Normal vector for placement orientation
            
            # Block outline for targeting visualization (single face)
            self.target_indicator = Entity(
                model='wireframe_quad',  # Single face outline
                color=color.rgb(0, 0, 0),  # Black outline
                scale=1.01,  # Slightly larger than block face
                enabled=False,  # Hidden until looking at a block
                always_on_top=True,  # Make sure it's visible through blocks
                thickness=8,  # Thicker lines
                double_sided=True
            )
            
            # Current block selection (for placement)
            self.selected_block_type = 'grass'  # Default block type
            
            # Block textures reference
            self.block_textures = block_type_textures
            
            # Hotbar
            self.hotbar = Hotbar(slot_count=6)
            self.selected_slot = 0
            
            # Block selection GUI
            self.block_gui = BlockGUI(self.hotbar, block_type_textures, player_ref=self)
            
            # Flying mode toggle (double-tap space)
            self.is_flying = True  # Start with flying enabled in Creative
            self.last_space_press_time = 0
            self.double_tap_window = 0.3  # 300ms window for double-tap
            self.air_time = 0  # Track time in air for gravity

        def update(self) -> None:  # pylint: disable=invalid-name
            if game_state["paused"]:
                return
            
            # Check for nearby dropped items to pick up
            self._check_item_pickup()
            
            # Handle flying mode toggle and gravity
            if not self.is_flying:
                # Walking mode - manually apply gravity since FirstPersonController doesn't toggle it at runtime
                self.gravity = 1
                
                # Apply gravity manually when not flying
                if not self.block_gui.visible:
                    # Check if player is on ground
                    ray = raycast(self.world_position + Vec3(0, 0.1, 0), direction=Vec3(0, -1, 0), distance=0.2, ignore=(self,), traverse_target=scene)
                    
                    if not ray.hit:
                        # Player is in air, apply gravity
                        self.air_time += time.dt
                        gravity_force = self.air_time * 5  # Accelerate downward
                        self.y -= gravity_force * time.dt
                    else:
                        # Player is on ground
                        self.air_time = 0
            else:
                # Flying mode - disable gravity
                self.gravity = 0
                self.air_time = 0
            
            # Prevent movement and camera rotation when block GUI is open
            if not self.block_gui.visible:
                # Let parent handle movement
                super().update()
                
                # Block targeting
                self._update_block_target()
            # GUI is open - movement and camera rotation disabled via mouse_sensitivity = 0
            
            # Set static block outline (no animation)
            if self.target_indicator.enabled:
                self.target_indicator.scale = Vec3(1.03, 1.03, 1.03)

            # Only allow vertical flying movement when GUI is closed AND in flying mode
            # This overrides the default space/shift behavior when flying
            if not self.block_gui.visible and self.is_flying:
                vertical_speed = self.flight_speed * time.dt
                if vertical_speed <= 0:
                    return

                ceiling_margin = 0.5
                floor_margin = 0.2
                safety_buffer = 0.1

                if held_keys["space"]:
                    hit = raycast(
                        origin=self.world_position + Vec3(0, self.eye_height - safety_buffer, 0),
                        direction=Vec3(0, 1, 0),
                        distance=vertical_speed + safety_buffer + 0.1,
                        ignore=(self,),
                        traverse_target=self.traverse_target,
                    )
                    if hit.hit:
                        ceiling_limit = hit.world_point.y - (self.eye_height + ceiling_margin)
                        if self.y > ceiling_limit:
                            self.y = ceiling_limit
                        else:
                            allowed = ceiling_limit - self.y
                            if allowed > 0:
                                self.y += min(vertical_speed, allowed)
                    else:
                        self.y += vertical_speed

                if held_keys["left shift"] or held_keys["shift"]:
                    hit = raycast(
                        origin=self.world_position + Vec3(0, 0.2, 0),
                        direction=Vec3(0, -1, 0),
                        distance=vertical_speed + 0.05,
                        ignore=(self,),
                        traverse_target=self.traverse_target,
                    )
                    if hit.hit:
                        floor_limit = hit.world_point.y + floor_margin
                        clearance = self.y - (floor_limit + safety_buffer)
                        if clearance <= 0:
                            self.y = max(self.y, floor_limit + safety_buffer)
                        else:
                            self.y -= min(vertical_speed, clearance)
                    else:
                        self.y -= vertical_speed

        def _update_block_target(self):
            """Raycast from player's camera to find block targets for interaction"""
            # Cast a ray from camera position in camera's forward direction
            hit_info = raycast(
                origin=camera.world_position,
                direction=camera.forward,
                distance=self.block_reach,
                traverse_target=scene,
                ignore=(self,),  # Ignore player entity
            )
            
            # Reset target information
            self.target_block = None
            self.target_position = None
            self.target_normal = None
            self.target_indicator.enabled = False
            
            if hit_info.hit:
                # Check if hit entity is a Block
                if isinstance(hit_info.entity, Block):
                    # Found a target block
                    self.target_block = hit_info.entity
                    
                    # Calculate placement position based on hit normal
                    self.target_normal = hit_info.normal
                    self.target_position = hit_info.entity.position + hit_info.normal
                    
                    # Show outline on the targeted face
                    self.target_indicator.enabled = True
                    # Offset the indicator slightly along the normal to place it on the face
                    face_offset = 0.501  # Half block size + small offset
                    self.target_indicator.position = hit_info.entity.position + (hit_info.normal * face_offset)
                    
                    # Rotate indicator to match the face orientation
                    if hit_info.normal.y != 0:  # Top or bottom face
                        self.target_indicator.rotation = Vec3(90 if hit_info.normal.y > 0 else -90, 0, 0)
                    elif hit_info.normal.x != 0:  # East or west face
                        self.target_indicator.rotation = Vec3(0, 90 if hit_info.normal.x > 0 else -90, 0)
                    else:  # North or south face
                        self.target_indicator.rotation = Vec3(0, 180 if hit_info.normal.z > 0 else 0, 0)
        
        def _check_item_pickup(self):
            """Check for nearby dropped items and pick them up."""
            player_pos = self.world_position
            pickup_range = 1.5  # Pickup range in blocks
            
            # Check all dropped items
            for item in dropped_items[:]:  # Use slice to avoid modification during iteration
                if item.try_pickup(player_pos, pickup_range):
                    # Add to an existing stack or first empty slot (up to max stack size)
                    added_slot = self.hotbar.add_item(item.block_type, max_stack=70)
                    if added_slot != -1:
                        logger.info(f"Picked up {item.block_type} into hotbar slot {added_slot + 1}")
                        # Remove item from world
                        item.pickup()
                        # Play pickup sound effect would go here
        
        def _update_camera_mode(self):
            """Update camera position and body visibility based on camera mode."""
            if self.camera_mode == 'first_person':
                # Hide body parts in first-person
                for part in self.body_parts.values():
                    part.visible = False
            else:  # third_person
                # Show body parts in third-person
                for part in self.body_parts.values():
                    part.visible = True
        
        def update(self) -> None:  # pylint: disable=invalid-name
            if game_state["paused"]:
                return
            
            # Check for nearby dropped items to pick up
            self._check_item_pickup()

            # Keep gravity in sync with flying state
            self.gravity = 0 if self.is_flying else 1
            
            # Handle vertical flight movement only when flying
            if self.is_flying:
                vertical_speed = self.flight_speed * time.dt
                
                if held_keys['space']:
                    # Check for ceiling collision before moving up
                    hit = raycast(
                        origin=self.world_position,
                        direction=(0, 1, 0),
                        distance=vertical_speed + self.height,
                        ignore=(self,),
                        traverse_target=scene
                    )
                    if not hit.hit:
                        self.y += vertical_speed
                
                if held_keys['shift']:
                    # Check for floor collision before moving down
                    hit = raycast(
                        origin=self.world_position,
                        direction=(0, -1, 0),
                        distance=vertical_speed + 0.1,  # Small buffer
                        ignore=(self,),
                        traverse_target=scene
                    )
                    if not hit.hit:
                        self.y -= vertical_speed
            
            # Call parent update for horizontal movement / jumping
            super().update()
            
            # Walking animation
            self._animate_walking()
            
            # Handle camera positioning based on mode
            if self.camera_mode == 'third_person':
                # Third-person: position camera behind player
                # Position camera at player eye level
                target_position = self.world_position + Vec3(0, self.eye_height, 0)
                
                # Move camera back from player's looking direction
                backward_offset = -camera.forward * self.third_person_distance
                camera.world_position = target_position + backward_offset
            # In first-person mode, let parent class handle camera position (for flight)
            
            # Update block targeting (works in both first and third person)
            self._update_block_target()
        
        def _animate_walking(self):
            """Animate arms and legs when walking (Minecraft-style)."""
            import math
            
            # Check if player is moving
            current_pos = Vec3(self.x, self.y, self.z)
            movement = (current_pos - self.last_position).length()
            self.is_moving = movement > 0.01
            self.last_position = current_pos
            
            if self.is_moving:
                # Update animation time when moving
                self.animation_time += time.dt * 10  # Animation speed
                
                # Calculate swing angles using sine wave
                swing_angle = 30  # Maximum swing angle in degrees
                arm_swing = math.sin(self.animation_time) * swing_angle
                leg_swing = math.sin(self.animation_time) * swing_angle
                
                # Animate arms (swing opposite to each other)
                if 'leftArm' in self.body_parts:
                    self.body_parts['leftArm'].rotation_x = arm_swing
                if 'rightArm' in self.body_parts:
                    self.body_parts['rightArm'].rotation_x = -arm_swing
                
                # Animate legs (swing opposite to arms and each other)
                if 'leftLeg' in self.body_parts:
                    self.body_parts['leftLeg'].rotation_x = -leg_swing
                if 'rightLeg' in self.body_parts:
                    self.body_parts['rightLeg'].rotation_x = leg_swing
            else:
                # Reset to neutral position when not moving
                if 'leftArm' in self.body_parts:
                    self.body_parts['leftArm'].rotation_x = 0
                if 'rightArm' in self.body_parts:
                    self.body_parts['rightArm'].rotation_x = 0
                if 'leftLeg' in self.body_parts:
                    self.body_parts['leftLeg'].rotation_x = 0
                if 'rightLeg' in self.body_parts:
                    self.body_parts['rightLeg'].rotation_x = 0
        
        def input(self, key):  # pylint: disable=invalid-name
            """Handle player input for block placement/breaking and GUI"""
            # Detect double-tap space to toggle flying
            if key == 'space':
                current_time = time.time()
                time_since_last_tap = current_time - self.last_space_press_time
                
                if time_since_last_tap < self.double_tap_window:
                    # Double-tap detected! Toggle flying
                    self.is_flying = not self.is_flying
                    if self.is_flying:
                        self.gravity = 0
                        logger.info("Flying mode enabled (gravity=0)")
                    else:
                        self.gravity = 1
                        logger.info(f"Flying mode disabled (gravity=1, position={self.world_position})")
                
                self.last_space_press_time = current_time
            
            # Block all input when GUI is open (except ESC and E to close)
            if self.block_gui.visible:
                if key == "escape" or key == 'e':
                    self.block_gui.hide()
                return  # Ignore all other inputs when GUI is open
            
            # F5 - Toggle camera mode (check before pause to allow toggling while paused)
            if key in ('f5', 'F5'):
                if self.camera_mode == 'first_person':
                    self.camera_mode = 'third_person'
                    logger.info("Switched to third-person camera")
                else:
                    self.camera_mode = 'first_person'
                    logger.info("Switched to first-person camera")
                self._update_camera_mode()
                return
            
            if key == "escape":
                # Close BlockGUI if open
                if self.block_gui.visible:
                    self.block_gui.hide()
                    return
                
                if game_state["paused"]:
                    resume_game()
                else:
                    pause_game()
                return

            if game_state["paused"]:
                return
            
            # Drop item (Q key)
            if key == 'q':
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    # Calculate drop position (in front of player)
                    drop_offset = camera.forward * 1.5  # 1.5 blocks in front
                    drop_position = self.world_position + Vec3(0, self.eye_height, 0) + drop_offset
                    
                    # Create dropped item entity
                    DroppedItem(
                        block_type=selected_item,
                        position=drop_position
                    )
                    
                    # Decrement stack count or clear slot if this was the last item
                    if hasattr(self.hotbar, "slot_counts") and self.selected_slot < len(self.hotbar.slot_counts):
                        if self.hotbar.slot_counts[self.selected_slot] > 1:
                            self.hotbar.slot_counts[self.selected_slot] -= 1
                            if hasattr(self.hotbar, "_update_slot_count_label"):
                                self.hotbar._update_slot_count_label(self.selected_slot)
                        else:
                            self.hotbar.set_slot_item(self.selected_slot, None)
                    else:
                        self.hotbar.set_slot_item(self.selected_slot, None)
                    self.hotbar.select_slot(self.selected_slot)  # Refresh display
                    
                    # If hotbar is now empty, clear selected block
                    if not self.hotbar.get_selected_item():
                        self.selected_block_type = None
                    
                    logger.info(f"Dropped {selected_item} item")
                return
            
            # Hotbar slot selection (1-6 keys)
            if key in '123456':
                slot_index = int(key) - 1
                self.selected_slot = slot_index
                self.hotbar.select_slot(slot_index)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                    logger.info(f"Selected {selected_item} block for placement")
                # Update global hotbar slot for persistence
                nonlocal current_hotbar_slot
                current_hotbar_slot = slot_index
                mark_world_dirty()
                return
            
            # Mouse wheel scrolling - cycle through hotbar slots with looping
            if key == 'scroll up':
                # Cycle backwards (loop from 0 to 5)
                self.selected_slot = (self.selected_slot - 1) % 6
                self.hotbar.select_slot(self.selected_slot)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                current_hotbar_slot = self.selected_slot
                mark_world_dirty()
                return
            
            if key == 'scroll down':
                # Cycle forwards (loop from 5 to 0)
                self.selected_slot = (self.selected_slot + 1) % 6
                self.hotbar.select_slot(self.selected_slot)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                current_hotbar_slot = self.selected_slot
                mark_world_dirty()
                return
            
            # 'E' key - Toggle block selection GUI
            if key == 'e':
                if self.block_gui.visible:
                    self.block_gui.hide()
                else:
                    self.block_gui.show()
                return

            # Call the parent input handler for other keys
            super().input(key)
            
            # Left mouse click - Break block
            if key == 'left mouse down':
                if self.target_block:
                    # Log the block breaking action
                    logger.info("Breaking block at %s", self.target_block.position)
                    
                    # Update adjacent blocks to show newly exposed faces
                    update_adjacent_blocks_on_removal(self.target_block.position)
                    
                    # Remove the block from the world
                    remove_block_from_state(self.target_block)
                    world_entities.remove(self.target_block)
                    destroy(self.target_block)
                    
                    # Reset targeting after breaking
                    self.target_block = None
                    self.target_indicator.enabled = False
            
            # Right mouse click - Place block
            elif key == 'right mouse down':
                if self.target_position:
                    # Get textures for the selected block type
                    textures = self.block_textures.get(self.selected_block_type)
                    if textures:
                        # Log the block placement action
                        logger.info("Placing %s block at %s", self.selected_block_type, self.target_position)
                        
                        # Create a new block
                        new_block = Block(
                            position=self.target_position,
                            block_size=1.0,
                            textures=textures,
                            block_type=self.selected_block_type,
                        )
                        new_block.name = f"{self.selected_block_type}_block_placed"
                        
                        # Update adjacent blocks to hide covered faces
                        update_adjacent_blocks_on_placement(self.target_position)
                        
                        # Visual feedback for block placement
                        place_effect = Entity(
                            model='wireframe_cube',
                            position=self.target_position,
                            scale=1.05,  # Slightly larger than the block
                            color=color.rgb(0, 0, 0),  # Black outline to match selection
                            thickness=2
                        )
                        # Make the effect briefly show then disappear
                        destroy(place_effect, delay=0.3)  # Slightly longer duration than break effect
                        
                        # Reset targeting after placement
                        self._update_block_target()
            
            # Old key selection removed - now handled by hotbar
    
    class SurvivalController(FirstPersonController):  # type: ignore[misc]
        def __init__(self, **kwargs):
            # Set height to exactly 2 blocks
            kwargs.setdefault("height", 2.0)
            super().__init__(**kwargs)
            self.speed = 5
            self.jump_height = 2.5

            # Eye height for camera and drop position (used when spawning dropped items)
            self.head_clearance = 0.5
            self.eye_height = max(self.height - self.head_clearance, 0.5)
            if hasattr(self, "camera_pivot"):
                self.camera_pivot.y = self.eye_height
            
            # Create player body model
            self.body_parts = create_player_body(self)
            logger.info("Created player model with %d body parts", len(self.body_parts))
            
            # Camera mode (first-person vs third-person)
            self.camera_mode = 'first_person'  # 'first_person' or 'third_person'
            self.third_person_distance = 5.0  # Distance behind player in third-person
            self._update_camera_mode()  # Initialize camera mode
            
            # Walking animation
            self.animation_time = 0
            self.last_position = Vec3(0, 0, 0)
            self.is_moving = False
            
            # Block interaction settings
            self.block_reach = 5.0
            self.target_block = None
            self.target_position = None
            self.target_normal = None
            
            # Block outline for targeting (single face)
            self.target_indicator = Entity(
                model='wireframe_quad',  # Single face outline
                color=color.rgb(0, 0, 0),
                scale=1.01,
                enabled=False,
                always_on_top=True,
                thickness=8,  # Thicker lines
                double_sided=True
            )
            
            # Current block selection
            self.selected_block_type = 'grass'
            self.block_textures = block_type_textures
            
            # Hotbar
            self.hotbar = Hotbar(slot_count=6)
            self.selected_slot = 0
            
            # Block selection GUI
            self.block_gui = BlockGUI(self.hotbar, block_type_textures, player_ref=self)
        
        def update(self):
            """Override update to add item pickup."""
            if game_state["paused"]:
                return
            
            # Check for nearby dropped items to pick up
            self._check_item_pickup()
            
            # Call parent update for movement
            super().update()
            
            # Update block targeting
            self._update_block_target()
            
        def _update_block_target(self):
            """Raycast to find block targets"""
            hit_info = raycast(
                origin=camera.world_position,
                direction=camera.forward,
                distance=self.block_reach,
                traverse_target=scene,
                ignore=(self,),
            )
            
            self.target_block = None
            self.target_position = None
            self.target_normal = None
            self.target_indicator.enabled = False
            
            if hit_info.hit and isinstance(hit_info.entity, Block):
                self.target_block = hit_info.entity
                self.target_normal = hit_info.normal
                self.target_position = hit_info.entity.position + hit_info.normal
                
                # Show outline on the targeted face
                self.target_indicator.enabled = True
                face_offset = 0.501
                self.target_indicator.position = hit_info.entity.position + (hit_info.normal * face_offset)
                
                # Rotate indicator to match the face orientation
                if hit_info.normal.y != 0:  # Top or bottom face
                    self.target_indicator.rotation = Vec3(90 if hit_info.normal.y > 0 else -90, 0, 0)
                elif hit_info.normal.x != 0:  # East or west face
                    self.target_indicator.rotation = Vec3(0, 90 if hit_info.normal.x > 0 else -90, 0)
                else:  # North or south face
                    self.target_indicator.rotation = Vec3(0, 180 if hit_info.normal.z > 0 else 0, 0)
        
        def _check_item_pickup(self):
            """Check for nearby dropped items and pick them up."""
            player_pos = self.world_position
            pickup_range = 1.5  # Pickup range in blocks
            
            # Check all dropped items
            for item in dropped_items[:]:  # Use slice to avoid modification during iteration
                if item.try_pickup(player_pos, pickup_range):
                    # Add to an existing stack or first empty slot (up to max stack size)
                    added_slot = self.hotbar.add_item(item.block_type, max_stack=70)
                    if added_slot != -1:
                        logger.info(f"Picked up {item.block_type} into hotbar slot {added_slot + 1}")
                        # Remove item from world
                        item.pickup()
                        # Play pickup sound effect would go here
        
        def _update_camera_mode(self):
            """Update camera position and body visibility based on camera mode."""
            if self.camera_mode == 'first_person':
                # Hide body parts in first-person
                for part in self.body_parts.values():
                    part.visible = False
            else:  # third_person
                # Show body parts in third-person
                for part in self.body_parts.values():
                    part.visible = True
        
        def update(self) -> None:
            if game_state["paused"]:
                return
            
            # Check for nearby dropped items to pick up
            self._check_item_pickup()
            
            # Call parent update for movement first
            super().update()
            
            # Walking animation
            self._animate_walking()
            
            # Get eye height (use attribute or calculate from height)
            eye_height = getattr(self, 'eye_height', self.height - 0.5)
            
            # Handle camera positioning based on mode
            if self.camera_mode == 'third_person':
                # Third-person: position camera behind player
                # Position camera at player eye level
                target_position = self.world_position + Vec3(0, eye_height, 0)
                
                # Move camera back from player's looking direction
                backward_offset = -camera.forward * self.third_person_distance
                camera.world_position = target_position + backward_offset
            # In first-person mode, let parent class handle camera position (for flight)
            
            # Update block targeting (works in both first and third person)
            self._update_block_target()
            if self.target_indicator.enabled:
                self.target_indicator.scale = Vec3(1.03, 1.03, 1.03)
        
        def _animate_walking(self):
            """Animate arms and legs when walking (Minecraft-style)."""
            import math
            
            # Check if player is moving
            current_pos = Vec3(self.x, self.y, self.z)
            movement = (current_pos - self.last_position).length()
            self.is_moving = movement > 0.01
            self.last_position = current_pos
            
            if self.is_moving:
                # Update animation time when moving
                self.animation_time += time.dt * 10  # Animation speed
                
                # Calculate swing angles using sine wave
                swing_angle = 30  # Maximum swing angle in degrees
                arm_swing = math.sin(self.animation_time) * swing_angle
                leg_swing = math.sin(self.animation_time) * swing_angle
                
                # Animate arms (swing opposite to each other)
                if 'leftArm' in self.body_parts:
                    self.body_parts['leftArm'].rotation_x = arm_swing
                if 'rightArm' in self.body_parts:
                    self.body_parts['rightArm'].rotation_x = -arm_swing
                
                # Animate legs (swing opposite to arms and each other)
                if 'leftLeg' in self.body_parts:
                    self.body_parts['leftLeg'].rotation_x = -leg_swing
                if 'rightLeg' in self.body_parts:
                    self.body_parts['rightLeg'].rotation_x = leg_swing
            else:
                # Reset to neutral position when not moving
                if 'leftArm' in self.body_parts:
                    self.body_parts['leftArm'].rotation_x = 0
                if 'rightArm' in self.body_parts:
                    self.body_parts['rightArm'].rotation_x = 0
                if 'leftLeg' in self.body_parts:
                    self.body_parts['leftLeg'].rotation_x = 0
                if 'rightLeg' in self.body_parts:
                    self.body_parts['rightLeg'].rotation_x = 0
        
        def input(self, key):
            """Handle input for survival mode"""
            # Block all input when GUI is open (except ESC and E to close)
            if self.block_gui.visible:
                if key == "escape" or key == 'e':
                    self.block_gui.hide()
                return  # Ignore all other inputs when GUI is open
            
            # F5 - Toggle camera mode (check before pause to allow toggling while paused)
            if key in ('f5', 'F5'):
                if self.camera_mode == 'first_person':
                    self.camera_mode = 'third_person'
                    logger.info("Switched to third-person camera")
                else:
                    self.camera_mode = 'first_person'
                    logger.info("Switched to first-person camera")
                self._update_camera_mode()
                return
            
            if key == "escape":
                # Close BlockGUI if open
                if self.block_gui.visible:
                    self.block_gui.hide()
                    return
                
                if game_state["paused"]:
                    resume_game()
                else:
                    pause_game()
                return

            if game_state["paused"]:
                return
            
            # Drop item (Q key)
            if key == 'q':
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    # Calculate drop position (in front of player)
                    drop_offset = camera.forward * 1.5  # 1.5 blocks in front
                    drop_position = self.world_position + Vec3(0, self.eye_height, 0) + drop_offset
                    
                    # Create dropped item entity
                    DroppedItem(
                        block_type=selected_item,
                        position=drop_position
                    )
                    
                    # Decrement stack count or clear slot if this was the last item
                    if hasattr(self.hotbar, "slot_counts") and self.selected_slot < len(self.hotbar.slot_counts):
                        if self.hotbar.slot_counts[self.selected_slot] > 1:
                            self.hotbar.slot_counts[self.selected_slot] -= 1
                            if hasattr(self.hotbar, "_update_slot_count_label"):
                                self.hotbar._update_slot_count_label(self.selected_slot)
                        else:
                            self.hotbar.set_slot_item(self.selected_slot, None)
                    else:
                        self.hotbar.set_slot_item(self.selected_slot, None)
                    self.hotbar.select_slot(self.selected_slot)  # Refresh display
                    
                    # If hotbar is now empty, clear selected block
                    if not self.hotbar.get_selected_item():
                        self.selected_block_type = None
                    
                    logger.info(f"Dropped {selected_item} item")
                return
            
            # Hotbar slot selection (1-6 keys)
            if key in '123456':
                slot_index = int(key) - 1
                self.selected_slot = slot_index
                self.hotbar.select_slot(slot_index)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                    logger.info(f"Selected {selected_item} block for placement")
                nonlocal current_hotbar_slot
                current_hotbar_slot = slot_index
                mark_world_dirty()
                return
            
            # Mouse wheel scrolling - cycle through hotbar slots with looping
            if key == 'scroll up':
                # Cycle backwards (loop from 0 to 5)
                self.selected_slot = (self.selected_slot - 1) % 6
                self.hotbar.select_slot(self.selected_slot)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                current_hotbar_slot = self.selected_slot
                mark_world_dirty()
                return
            
            if key == 'scroll down':
                # Cycle forwards (loop from 5 to 0)
                self.selected_slot = (self.selected_slot + 1) % 6
                self.hotbar.select_slot(self.selected_slot)
                selected_item = self.hotbar.get_selected_item()
                if selected_item:
                    self.selected_block_type = selected_item
                current_hotbar_slot = self.selected_slot
                mark_world_dirty()
                return
            
            # 'E' key - Toggle block selection GUI
            if key == 'e':
                if self.block_gui.visible:
                    self.block_gui.hide()
                else:
                    self.block_gui.show()
                return

            super().input(key)
            
            # Left mouse - Break block
            if key == 'left mouse down':
                if self.target_block:
                    logger.info("Breaking block at %s", self.target_block.position)
                    # Update adjacent blocks to show newly exposed faces
                    update_adjacent_blocks_on_removal(self.target_block.position)
                    remove_block_from_state(self.target_block)
                    world_entities.remove(self.target_block)
                    destroy(self.target_block)
            
            # Right mouse - Place block
            if key == 'right mouse down':
                if self.target_position is not None:
                    rounded_position = Vec3(
                        round(self.target_position.x),
                        round(self.target_position.y),
                        round(self.target_position.z)
                    )
                    
                    occupied = any(
                        e.position == rounded_position
                        for e in world_entities
                        if isinstance(e, Block)
                    )
                    
                    if not occupied:
                        textures = self.block_textures.get(self.selected_block_type, face_textures)
                        logger.info("Placing %s block at %s", self.selected_block_type, rounded_position)
                        new_block = Block(
                            position=rounded_position,
                            block_size=1.0,
                            textures=textures,
                            block_type=self.selected_block_type,
                        )
                        new_block.collision = True
                        
                        # Update adjacent blocks to hide covered faces
                        update_adjacent_blocks_on_placement(rounded_position)
            

    # Load dirt block textures
    def load_dirt_textures():
        dirt_face_paths = {
            "north": "blocks/textures/dirt_side.png",
            "south": "blocks/textures/dirt_side.png",
            "east": "blocks/textures/dirt_side.png",
            "west": "blocks/textures/dirt_side.png",
            "up": "blocks/textures/dirt_top.png",
            "down": "blocks/textures/dirt_bottom.png",
        }

        dirt_textures: Dict[str, object] = {}
        for face, rel_path in dirt_face_paths.items():
            texture = load_texture(rel_path)
            if texture:
                logger.info("Loaded dirt texture for %s: %s", face, rel_path)
                texture.filtering = False  # Disable filtering for crisp pixel art
                dirt_textures[face] = texture
            else:
                logger.warning("Failed to load dirt texture for %s: %s", face, rel_path)
                # Fall back to regular textures
                dirt_textures[face] = face_textures.get(face)
        return dirt_textures

    # Initialize dirt textures
    dirt_textures = load_dirt_textures()
    block_type_textures["dirt"] = dirt_textures

    def generate_flat_world(width: int, depth: int, block_size: float, *, mark_dirty: bool = True) -> None:
        nonlocal enable_face_culling
        logger.info("Generating flat world with %dx%d blocks", width, depth)

        # Disable face culling during world generation
        enable_face_culling = False
        
        # Reset tracking structures
        current_world_blocks.clear()
        if not mark_dirty:
            reset_dirty_flag()

        # Center the world around (0,0,0)
        start_x = -((width - 1) * block_size) / 2
        start_z = -((depth - 1) * block_size) / 2

        # Track block counts for debugging
        total_blocks = 0

        # Resolve textures for block types
        grass_textures = block_type_textures.get("grass", face_textures)
        dirt_textures_local = block_type_textures.get("dirt", face_textures)

        # Create a 3-layer world: 1 grass layer on top of 2 dirt layers
        logger.info("Starting block placement, total expected blocks: %d", width * depth * 3)  # 3 layers
        for ix in range(width):
            for iz in range(depth):
                # Grass block (top layer)
                position = Vec3(
                    start_x + ix * block_size,
                    block_size * 2.5,  # At y=2.5 (third layer)
                    start_z + iz * block_size,
                )
                grass_block = Block(
                    position=position,
                    block_size=block_size,
                    textures=grass_textures,
                    block_type="grass",
                    mark_dirty=mark_dirty,
                )
                grass_block.name = f"grass_block_{ix}_{iz}"
                grass_block.collision = True
                total_blocks += 1

                # Dirt blocks (2 layers underneath)
                for y in range(2):  # Create 2 layers of dirt
                    dirt_position = Vec3(
                        start_x + ix * block_size,
                        block_size * (y + 0.5),  # At y=0.5 and y=1.5
                        start_z + iz * block_size,
                    )
                    dirt_block = Block(
                        position=dirt_position,
                        block_size=block_size,
                        textures=dirt_textures_local,
                        block_type="dirt",
                        mark_dirty=mark_dirty,
                    )
                    dirt_block.name = f"dirt_block_{ix}_{iz}_{y}"
                    dirt_block.collision = True
                    total_blocks += 1

        # Enable face culling after generation
        enable_face_culling = True
        logger.info("World generation complete: placed %d blocks", total_blocks)

    def clear_world() -> None:
        nonlocal player_entity, current_player_position, current_player_orientation, current_hotbar_slot
        for entity in world_entities[:]:
            destroy(entity)
            world_entities.remove(entity)
        if player_entity is not None:
            # Destroy hotbar when leaving world
            if hasattr(player_entity, 'hotbar') and player_entity.hotbar:
                destroy(player_entity.hotbar)
            destroy(player_entity)
            player_entity = None
        current_world_blocks.clear()
        current_player_position = None
        current_player_orientation = None

    menu_textures = {
        "backgrounds": {
            "main": load_texture("menu/backgrounds/main_bg.png"),
            "singleplayer": load_texture("menu/backgrounds/singleplayer_bg.png"),
            "multiplayer": load_texture("menu/backgrounds/multiplayer_bg.png"),
            "options": load_texture("menu/backgrounds/options_bg.png"),
        },
        "buttons": {
            "singleplayer": load_texture("menu/buttons/singleplayer.png"),
            "multiplayer": load_texture("menu/buttons/Multiplayer.png"),
            "options": load_texture("menu/buttons/Options.png"),
            "quit": load_texture("menu/buttons/quit.png"),
            "create_world": load_texture("menu/buttons/create_world.png"),
            "back": load_texture("menu/buttons/singleplayer_back.png"),
            "create_world_small": load_texture("menu/buttons/Create_world_smaller.png"),
            "cancel": load_texture("menu/buttons/cancel.png"),
            "survival": load_texture("menu/buttons/Survive.png"),
            "sandbox": load_texture("menu/buttons/sandbox.png"),
            "delete_world": load_texture("menu/buttons/delete_world.png"),
            "done": load_texture("menu/buttons/apply_button.png"),
        },
        "panels": {
            "world_list": load_texture("menu/GUIS/Gui_grey.png"),
            "dialog": load_texture("menu/GUIS/Gui_grey.png"),
        },
    }

    for category in menu_textures.values():
        for texture in category.values():
            if texture is not None:
                texture.alpha = "blend"
                texture.mipmap = False

    class MenuManager:
        def __init__(self, on_start_game):
            self.on_start_game = on_start_game
            self.root = Entity(parent=camera.ui)
            self.background = None
            self.widgets: list[Entity] = []
            self.state = "main"
            self.info_text: Optional[Text] = None
            self.dialog_root: Optional[Entity] = None
            self.world_name_input: Optional[InputField] = None
            self.world_list_container: Optional[Entity] = None
            self.world_entries: list[Entity] = []
            self.available_worlds: list[Dict[str, Any]] = []
            self.confirmation_root: Optional[Entity] = None
            self.pending_world_info: Optional[Dict[str, Any]] = None
            self.show_main_menu()

        def _show_confirmation_dialog(
            self,
            message: str,
            on_confirm,
            confirm_label: str = "OK",
            cancel_label: str = "Cancel",
        ) -> None:
            if hasattr(self, "confirmation_root") and self.confirmation_root is not None:
                destroy(self.confirmation_root)

            overlay = Entity(parent=camera.ui, z=-5)
            bg = Entity(parent=overlay, model="quad", color=color.rgba(0, 0, 0, 180), scale=(camera.aspect_ratio * 2, 2))

            panel = Entity(
                parent=overlay,
                model="quad",
                color=color.rgb(45, 45, 60),  # Slightly lighter background for better contrast
                scale=(0.6, 0.5),  # Make panel taller to fit everything
            )

            # Dialog title in bigger text
            title_text = Text(
                parent=panel,
                text="DELETE WORLD",  # Always show as delete world title
                origin=(0, 0),
                position=(0, 0.15),
                scale=1.6,
                color=color.rgb(255, 255, 255),
            )
            title_text.thickness = 3
            
            # World name in slightly smaller text
            world_name = message.strip("'")  # We now directly pass the world name in the message
            name_text = Text(
                parent=panel,
                text=world_name,
                origin=(0, 0),
                position=(0, 0.05),
                scale=1.4,
                color=color.rgb(255, 255, 255),
            )
            name_text.thickness = 2
            
            # Description text at the bottom
            desc_text = Text(
                parent=panel,
                text="This world will be moved to your Recycle Bin.",
                origin=(0, 0),
                position=(0, -0.05),
                scale=1.2,
                color=color.rgb(200, 200, 200),  # Slightly dimmer
                wordwrap=30,
            )

            def cleanup():
                if overlay:
                    destroy(overlay)
                    self.confirmation_root = None

            def confirm_action():
                try:
                    on_confirm()
                finally:
                    cleanup()

            def cancel_action():
                cleanup()

            # Simple Delete button
            confirm_button = Button(
                parent=panel,
                text="Delete",  # Fixed text
                color=color.rgba(180, 60, 60, 255),  # Red for delete
                highlight_color=color.rgba(220, 80, 80, 255),
                pressed_color=color.rgba(150, 40, 40, 255),
                scale=(0.2, 0.1),  # Smaller, more compact
                position=(-0.15, -0.18),  # Lower position
            )
            confirm_button.on_click = confirm_action

            # Simple Cancel button 
            cancel_button = Button(
                parent=panel,
                text="Cancel",  # Fixed text
                color=color.rgba(70, 70, 90, 255),  # Dark gray
                highlight_color=color.rgba(90, 90, 120, 255),
                pressed_color=color.rgba(50, 50, 70, 255),
                scale=(0.2, 0.1),  # Smaller, more compact
                position=(0.15, -0.18),  # Lower position
            )
            cancel_button.on_click = cancel_action

            self.confirmation_root = overlay

        def clear_widgets(self) -> None:
            for widget in self.widgets:
                destroy(widget)
            self.widgets.clear()
            if self.info_text is not None:
                destroy(self.info_text)
                self.info_text = None
            for entry in self.world_entries:
                destroy(entry)
            self.world_entries.clear()
            if self.world_list_container is not None:
                destroy(self.world_list_container)
                self.world_list_container = None

        def close_dialog(self) -> None:
            if self.dialog_root is not None:
                destroy(self.dialog_root)
                self.dialog_root = None
            self.world_name_input = None

        def render_world_list(self) -> None:
            if self.world_list_container is not None:
                destroy(self.world_list_container)
            self.world_entries.clear()
            self.world_list_container = Entity(parent=self.root)

            panel = Entity(
                parent=self.world_list_container,
                model="quad",
                texture=menu_textures["panels"].get("world_list"),
                scale=(0.9, 0.55),
                position=(0, 0.2),
            )
            self.world_entries.append(panel)

            if not self.available_worlds:
                empty_text = Text(
                    parent=self.world_list_container,
                    text="No worlds found",
                    origin=(0, 0),
                    position=(0, 0.3),
                    scale=1,
                    color=color.white,
                )
                self.world_entries.append(empty_text)
                return

            y_offset = 0.28
            max_display_chars = 22
            for world_info in self.available_worlds[:6]:
                entry_button = Button(
                    parent=self.world_list_container,
                    text="",
                    model="quad",
                    color=color.rgba(40, 40, 60, 200),
                    highlight_color=color.rgba(80, 80, 120, 220),
                    pressed_color=color.rgba(30, 30, 50, 230),
                    scale=(0.65, 0.11),
                    position=(0.0, y_offset),
                    origin=(0, 0),
                )
                entry_button.on_click = lambda info=world_info: self.launch_world(info)
                self.world_entries.append(entry_button)

                full_name = world_info["display_name"]
                display_text = full_name
                if len(display_text) > max_display_chars:
                    display_text = display_text[: max_display_chars - 1] + "…"

                # Use fixed text size values for consistent results
                # This approach avoids scaling issues by using specific values
                button_width = entry_button.scale_x  # Define for logging
                button_height = entry_button.scale_y  # Define for logging
                text_scale = (4.0, 10.0)  # Increased overall scale - wider and taller

                logger.info(
                    "Rendering world entry: original='%s', display='%s', button=%sx%s, text_scale=%s",
                    full_name,
                    display_text,
                    button_width, 
                    button_height,
                    text_scale,
                )

                name_label = Text(
                    parent=entry_button,
                    text=display_text,
                    origin=(0, 0),
                    position=(0, 0),
                    scale=text_scale,
                    color=color.rgb(255, 255, 255),
                    wordwrap=0,
                )
                name_label.thickness = 14  # Even bolder text
                name_label.z = -0.01
                self.world_entries.append(name_label)

                delete_x = entry_button.position.x + entry_button.scale_x / 2 + 0.045
                delete_button = Button(
                    parent=self.world_list_container,
                    model="quad",
                    texture=menu_textures["buttons"].get("delete_world"),
                    scale=(0.08, 0.08),
                    position=(delete_x, y_offset),
                    origin=(0, 0),
                    color=color.white,
                    highlight_color=color.rgb(255, 200, 200),
                    pressed_color=color.rgb(220, 100, 100),
                )

                def prompt_delete(info=world_info):
                    display_name = info["display_name"]

                    def do_delete() -> None:
                        try:
                            send2trash(str(info["path"]))
                            logger.info("Moved world '%s' to recycle bin", display_name)
                            self.show_message(f"Moved '{display_name}' to Recycle Bin.")
                        except Exception as exc:
                            logger.exception("Failed to delete world '%s': %s", display_name, exc)
                            self.show_message(f"Failed to delete '{display_name}'. See log for details.")
                        finally:
                            self.refresh_worlds()

                    self._show_confirmation_dialog(
                        message=f"'{display_name}'",  # Only pass the world name - other text is now hardcoded in the dialog
                        
                        on_confirm=do_delete,
                        confirm_label="Delete",
                        cancel_label="Cancel",
                    )

                delete_button.on_click = prompt_delete
                self.world_entries.append(delete_button)

                y_offset -= 0.12

        def refresh_worlds(self) -> None:
            self.available_worlds = list_worlds()
            self.render_world_list()

        def launch_world(self, world_info: Dict[str, Any]) -> None:
            metadata = world_info["metadata"]
            metadata["last_played_at"] = datetime.utcnow().isoformat(timespec="seconds")
            save_world_metadata(world_info["path"], metadata)
            world_info["metadata"] = metadata
            logger.info("Launching world '%s'", metadata.get("display_name", world_info["path"].name))
            self.selected_mode = metadata.get("mode", "creative")
            self.pending_world_info = world_info
            self.start_game(world_info)

        def set_background(self, texture_name: str) -> None:
            texture = menu_textures["backgrounds"].get(texture_name)
            if self.background is None:
                self.background = Entity(parent=self.root, model="quad")
            self.background.texture = texture
            self.background.scale = (window.aspect_ratio * 2, 2)
            self.background.z = 1

        def _make_button(self, texture_key: str, position: tuple[float, float], on_click) -> Button:
            button_texture = menu_textures["buttons"].get(texture_key)

            def safe_click():
                try:
                    on_click()
                except Exception as exc:
                    logger.exception("Unhandled error in menu action: %s", exc)
                    raise

            button = Button(
                parent=self.root,
                model="quad",
                texture=button_texture,
                scale=(0.6, 0.16),
                position=position,
                origin=(0, 0),
                color=color.white,
                highlight_color=color.white,
                pressed_color=color.white,
                text="",
            )
            button.on_click = safe_click
            self.widgets.append(button)
            return button

        def show_main_menu(self) -> None:
            self.state = "main"
            self.root.enable()
            self.set_background("main")
            self.clear_widgets()

            def go_singleplayer():
                logger.info("Menu: Singleplayer selected")
                self.show_singleplayer_menu()

            def go_multiplayer():
                logger.info("Menu: Multiplayer selected (not implemented)")
                self.show_message("Multiplayer is not available yet.")

            def go_options():
                logger.info("Menu: Options selected")
                self.show_settings_menu()

            def quit_game():
                logger.info("Menu: Quit selected")
                save_current_world_state(force=True)
                clear_world()
                application.quit()

            self._make_button("singleplayer", (0, 0.25), go_singleplayer)
            self._make_button("multiplayer", (0, 0.05), go_multiplayer)
            self._make_button("options", (0, -0.15), go_options)
            self._make_button("quit", (0, -0.35), quit_game)

        def show_singleplayer_menu(self) -> None:
            self.state = "singleplayer"
            self.set_background("singleplayer")
            self.clear_widgets()

            def go_back():
                logger.info("Menu: Back to main")
                self.show_main_menu()

            def create_world():
                logger.info("Menu: Create World selected")
                self.show_create_world_dialog()

            world_panel = Entity(
                parent=self.root,
                model="quad",
                texture=menu_textures["panels"].get("world_list"),
                scale=(0.9, 0.55),
                position=(0, 0.2),
            )
            self.widgets.append(world_panel)

            self.available_worlds = list_worlds()
            self.render_world_list()

            # Create World button (smaller and positioned higher to fit screen)
            create_world_texture = menu_textures["buttons"].get("create_world")
            create_world_btn = Button(
                parent=self.root,
                model="quad",
                texture=create_world_texture,
                scale=(0.45, 0.12),  # Smaller than default
                position=(0, -0.18),  # Moved higher to fit screen
                origin=(0, 0),
                color=color.white,
                highlight_color=color.white,
                pressed_color=color.white,
            )
            
            def safe_create_world():
                try:
                    create_world()
                except Exception as exc:
                    logger.exception("Unhandled error in menu action: %s", exc)
                    raise
            
            create_world_btn.on_click = safe_create_world
            self.widgets.append(create_world_btn)
            
            # Back button (positioned below Create World to fit screen)
            self._make_button("back", (0, -0.33), go_back)

            self.show_message("Select a world or create a new one.")
        
        def show_settings_menu(self) -> None:
            """Display interactive settings menu with sliders and controls."""
            self.state = "settings"
            self.set_background("main")
            self.clear_widgets()
            
            # Load current settings
            settings = load_settings()
            
            # Title
            title = Text(
                parent=self.root,
                text="Settings",
                origin=(0, 0),
                position=(0, 0.45),
                scale=2.5,
                color=color.white,
            )
            title.thickness = 4
            self.widgets.append(title)
            
            # Resolution dropdown
            res_label = Text(
                parent=self.root,
                text="Resolution",
                origin=(0, 0),
                position=(-0.35, 0.32),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(res_label)
            
            monitor_width, monitor_height = get_monitor_resolution()
            resolution_options = [
                "auto",
                f"{monitor_width}x{monitor_height}",
                "1920x1080",
                "1680x1050",
                "1600x900",
                "1366x768",
                "1280x720",
            ]
            current_res = settings["resolution"]
            res_index = resolution_options.index(current_res) if current_res in resolution_options else 0
            res_state = {"index": res_index, "options": resolution_options, "dropdown_open": False, "dropdown_widgets": []}
            
            # Create main dropdown button showing current selection
            res_button_bg = Button(
                parent=self.root,
                model='quad',
                color=color.rgb(60, 60, 80),
                highlight_color=color.rgb(80, 80, 100),
                scale=(0.45, 0.065),
                position=(0.15, 0.32),
                text="",
            )
            self.widgets.append(res_button_bg)
            
            res_text = Text(
                parent=self.root,
                text=resolution_options[res_index],
                position=(0.15, 0.32),
                origin=(0, 0),
                scale=1.3,
                color=color.white,
            )
            res_text.z = -0.1  # Place text in front
            self.widgets.append(res_text)
            
            def toggle_dropdown():
                if res_state["dropdown_open"]:
                    # Close dropdown
                    for widget in res_state["dropdown_widgets"]:
                        destroy(widget)
                    res_state["dropdown_widgets"] = []
                    res_state["dropdown_open"] = False
                else:
                    # Open dropdown - show all options
                    res_state["dropdown_open"] = True
                    y_pos = 0.25  # Start below the button
                    
                    for i, option in enumerate(resolution_options):
                        opt_bg = Entity(
                            parent=self.root,
                            model='quad',
                            color=color.rgb(50, 50, 70) if i != res_state["index"] else color.rgb(70, 100, 130),
                            scale=(0.45, 0.055),
                            position=(0.15, y_pos),
                            z=-0.05,
                        )
                        
                        opt_text = Text(
                            parent=self.root,
                            text=option,
                            position=(0.15, y_pos),
                            origin=(0, 0),
                            scale=1.2,
                            color=color.white,
                            z=-0.1,
                        )
                        
                        opt_button = Button(
                            parent=self.root,
                            model='quad',
                            color=color.rgba(0, 0, 0, 0),  # Transparent
                            scale=(0.45, 0.055),
                            position=(0.15, y_pos),
                            z=-0.15,
                        )
                        
                        def make_selector(index, opt):
                            def select():
                                res_state["index"] = index
                                res_text.text = opt
                                # Close dropdown
                                for widget in res_state["dropdown_widgets"]:
                                    destroy(widget)
                                res_state["dropdown_widgets"] = []
                                res_state["dropdown_open"] = False
                            return select
                        
                        opt_button.on_click = make_selector(i, option)
                        
                        res_state["dropdown_widgets"].extend([opt_bg, opt_text, opt_button])
                        y_pos -= 0.06
            
            res_button_bg.on_click = toggle_dropdown
            
            self.widgets.append(res_text)
            
            # Fullscreen toggle
            fullscreen_label = Text(
                parent=self.root,
                text="Fullscreen",
                origin=(0, 0),
                position=(-0.35, 0.22),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(fullscreen_label)
            
            fullscreen_button = Button(
                parent=self.root,
                text="ON" if settings["fullscreen"] else "OFF",
                color=color.rgb(60, 180, 60) if settings["fullscreen"] else color.rgb(180, 60, 60),
                scale=(0.15, 0.06),
                position=(0.15, 0.22),
            )
            fullscreen_state = {"value": settings["fullscreen"]}
            
            def toggle_fullscreen():
                fullscreen_state["value"] = not fullscreen_state["value"]
                fullscreen_button.text = "ON" if fullscreen_state["value"] else "OFF"
                fullscreen_button.color = color.rgb(60, 180, 60) if fullscreen_state["value"] else color.rgb(180, 60, 60)
            
            fullscreen_button.on_click = toggle_fullscreen
            self.widgets.append(fullscreen_button)
            
            # V-Sync toggle
            vsync_label = Text(
                parent=self.root,
                text="V-Sync",
                origin=(0, 0),
                position=(-0.35, 0.12),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(vsync_label)
            
            vsync_button = Button(
                parent=self.root,
                text="ON" if settings["vsync"] else "OFF",
                color=color.rgb(60, 180, 60) if settings["vsync"] else color.rgb(180, 60, 60),
                scale=(0.15, 0.06),
                position=(0.15, 0.12),
            )
            vsync_state = {"value": settings["vsync"]}
            
            def toggle_vsync():
                vsync_state["value"] = not vsync_state["value"]
                vsync_button.text = "ON" if vsync_state["value"] else "OFF"
                vsync_button.color = color.rgb(60, 180, 60) if vsync_state["value"] else color.rgb(180, 60, 60)
            
            vsync_button.on_click = toggle_vsync
            self.widgets.append(vsync_button)
            
            # FOV slider
            fov_label = Text(
                parent=self.root,
                text=f"FOV: {settings['fov']}°",
                origin=(0, 0),
                position=(-0.35, 0.0),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(fov_label)
            
            fov_slider = Slider(
                min=60, max=120, default=settings['fov'], step=5,
                parent=self.root,
                position=(0.15, 0.0),
                width=0.3,
                height=0.04,
                dynamic=True
            )
            
            def update_fov():
                fov_label.text = f"FOV: {int(fov_slider.value)}°"
                # Apply FOV change immediately
                camera.fov = int(fov_slider.value)
            
            fov_slider.on_value_changed = update_fov
            self.widgets.append(fov_slider)
            
            # Sound volume slider
            sound_label = Text(
                parent=self.root,
                text=f"Sound: {int(settings['sound_volume'] * 100)}%",
                origin=(0, 0),
                position=(-0.35, -0.12),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(sound_label)
            
            sound_slider = Slider(
                min=0, max=1, default=settings['sound_volume'], step=0.05,
                parent=self.root,
                position=(0.15, -0.12),
                width=0.3,
                height=0.04,
                dynamic=True
            )
            
            def update_sound():
                sound_label.text = f"Sound: {int(sound_slider.value * 100)}%"
            
            sound_slider.on_value_changed = update_sound
            self.widgets.append(sound_slider)
            
            # Music volume slider
            music_label = Text(
                parent=self.root,
                text=f"Music: {int(settings['music_volume'] * 100)}%",
                origin=(0, 0),
                position=(-0.35, -0.24),
                scale=1.2,
                color=color.white,
            )
            self.widgets.append(music_label)
            
            music_slider = Slider(
                min=0, max=1, default=settings['music_volume'], step=0.05,
                parent=self.root,
                position=(0.15, -0.24),
                width=0.3,
                height=0.04,
                dynamic=True
            )
            
            def update_music():
                music_label.text = f"Music: {int(music_slider.value * 100)}%"
            
            music_slider.on_value_changed = update_music
            self.widgets.append(music_slider)
            
            # Save & Back buttons
            def save_and_back():
                # Check if native resolution is selected in windowed mode - auto-enable fullscreen
                selected_res = resolution_options[res_state["index"]]
                if selected_res != "auto" and not fullscreen_state["value"]:
                    try:
                        check_width, check_height = map(int, selected_res.split("x"))
                        monitor_w, monitor_h = get_monitor_resolution()
                        if check_width >= monitor_w and check_height >= monitor_h:
                            logger.info(f"Native resolution {selected_res} selected - auto-enabling fullscreen")
                            fullscreen_state["value"] = True
                            fullscreen_button.text = "ON"
                            fullscreen_button.color = color.rgb(60, 180, 60)
                    except:
                        pass
                
                # Gather all settings
                new_settings = {
                    "resolution": selected_res,
                    "refresh_rate": settings["refresh_rate"],  # Not changed in UI yet
                    "fullscreen": fullscreen_state["value"],
                    "vsync": vsync_state["value"],
                    "fov": int(fov_slider.value),
                    "sound_volume": round(sound_slider.value, 2),
                    "music_volume": round(music_slider.value, 2),
                }
                
                # Save settings
                save_settings(new_settings)
                logger.info(f"Settings saved: {new_settings}")
                
                # Apply ALL settings immediately
                settings_applied = []
                
                try:
                    from panda3d.core import WindowProperties, loadPrcFileData
                    from direct.showbase.ShowBase import ShowBase
                    
                    # 1. Apply resolution/fullscreen changes
                    if new_settings["resolution"] != settings["resolution"] or new_settings["fullscreen"] != settings["fullscreen"]:
                        props = WindowProperties()
                        
                        # Calculate new window size
                        if new_settings["resolution"] == "auto":
                            monitor_w, monitor_h = get_monitor_resolution()
                            if not new_settings["fullscreen"]:
                                new_width = int(monitor_w * 0.8)
                                new_height = int(monitor_h * 0.8)
                            else:
                                new_width = monitor_w
                                new_height = monitor_h
                        else:
                            try:
                                new_width, new_height = map(int, new_settings["resolution"].split("x"))
                            except:
                                new_width, new_height = 1920, 1080
                        
                        # Apply window changes - try multiple methods
                        # For fullscreen mode at non-native resolution, change display mode
                        if new_settings["fullscreen"]:
                            monitor_w, monitor_h = get_monitor_resolution()
                            logger.info(f"Fullscreen mode: checking {new_width}x{new_height} against native {monitor_w}x{monitor_h}")
                            
                            # If resolution is smaller than native, change the display mode
                            if new_width < monitor_w or new_height < monitor_h:
                                try:
                                    import ctypes
                                    from ctypes import wintypes
                                    
                                    # Define DEVMODE structure for display mode change
                                    class DEVMODE(ctypes.Structure):
                                        _fields_ = [
                                            ('dmDeviceName', ctypes.c_wchar * 32),
                                            ('dmSpecVersion', ctypes.c_ushort),
                                            ('dmDriverVersion', ctypes.c_ushort),
                                            ('dmSize', ctypes.c_ushort),
                                            ('dmDriverExtra', ctypes.c_ushort),
                                            ('dmFields', ctypes.c_ulong),
                                            ('dmPositionX', ctypes.c_long),
                                            ('dmPositionY', ctypes.c_long),
                                            ('dmDisplayOrientation', ctypes.c_ulong),
                                            ('dmDisplayFixedOutput', ctypes.c_ulong),
                                            ('dmColor', ctypes.c_short),
                                            ('dmDuplex', ctypes.c_short),
                                            ('dmYResolution', ctypes.c_short),
                                            ('dmTTOption', ctypes.c_short),
                                            ('dmCollate', ctypes.c_short),
                                            ('dmFormName', ctypes.c_wchar * 32),
                                            ('dmLogPixels', ctypes.c_ushort),
                                            ('dmBitsPerPel', ctypes.c_ulong),
                                            ('dmPelsWidth', ctypes.c_ulong),
                                            ('dmPelsHeight', ctypes.c_ulong),
                                            ('dmDisplayFlags', ctypes.c_ulong),
                                            ('dmDisplayFrequency', ctypes.c_ulong),
                                            ('dmICMMethod', ctypes.c_ulong),
                                            ('dmICMIntent', ctypes.c_ulong),
                                            ('dmMediaType', ctypes.c_ulong),
                                            ('dmDitherType', ctypes.c_ulong),
                                            ('dmReserved1', ctypes.c_ulong),
                                            ('dmReserved2', ctypes.c_ulong),
                                            ('dmPanningWidth', ctypes.c_ulong),
                                            ('dmPanningHeight', ctypes.c_ulong),
                                        ]
                                    
                                    # Change display mode to the requested resolution
                                    dm = DEVMODE()
                                    dm.dmSize = ctypes.sizeof(DEVMODE)
                                    dm.dmPelsWidth = new_width
                                    dm.dmPelsHeight = new_height
                                    dm.dmFields = 0x180000  # DM_PELSWIDTH | DM_PELSHEIGHT
                                    
                                    user32 = ctypes.windll.user32
                                    CDS_FULLSCREEN = 0x00000004
                                    result = user32.ChangeDisplaySettingsW(ctypes.byref(dm), CDS_FULLSCREEN)
                                    
                                    if result == 0:  # DISP_CHANGE_SUCCESSFUL
                                        logger.info(f"Changed display mode to {new_width}x{new_height}")
                                    else:
                                        logger.warning(f"Failed to change display mode, result code: {result}")
                                except Exception as e:
                                    logger.error(f"Error changing display mode: {e}")
                                
                                # Now set window to fullscreen at this resolution
                                props.setSize(new_width, new_height)
                                props.setFullscreen(True)
                                props.setOrigin(0, 0)
                                props.setUndecorated(True)
                            else:
                                # Native or higher resolution - normal fullscreen
                                props.setSize(new_width, new_height)
                                props.setFullscreen(True)
                        else:
                            # Windowed mode - restore native resolution if it was changed
                            try:
                                import ctypes
                                user32 = ctypes.windll.user32
                                # Passing None restores the default display mode
                                user32.ChangeDisplaySettingsW(None, 0)
                                logger.info("Restored default display mode for windowed mode")
                            except:
                                pass
                            
                            props.setSize(new_width, new_height)
                            props.setFullscreen(False)
                            props.setUndecorated(False)
                        
                        # Method 1: Try window.entity (Ursina way)
                        applied = False
                        if hasattr(window, 'entity') and window.entity:
                            window.entity.requestProperties(props)
                            applied = True
                        
                        # Method 2: Try base.win (Panda3D way)
                        if not applied:
                            try:
                                import builtins
                                if hasattr(builtins, 'base') and hasattr(builtins.base, 'win'):
                                    builtins.base.win.requestProperties(props)
                                    applied = True
                            except:
                                pass
                        
                        # Method 3: Direct window manipulation (Ursina)
                        if not applied:
                            try:
                                window.size = (new_width, new_height)
                                window.fullscreen = new_settings["fullscreen"]
                                applied = True
                            except:
                                pass
                        
                        if applied:
                            # Update aspect ratio for correct mouse coordinates
                            window.aspect_ratio = new_width / new_height
                            
                            if new_settings["fullscreen"] != settings["fullscreen"]:
                                settings_applied.append("resolution + fullscreen")
                            else:
                                settings_applied.append("resolution")
                            logger.info(f"Applied resolution: {new_width}x{new_height}, Fullscreen: {new_settings['fullscreen']}")
                        else:
                            logger.warning("Could not apply resolution changes")
                    
                    # 2. Apply V-Sync changes immediately
                    if new_settings["vsync"] != settings["vsync"]:
                        loadPrcFileData('', f'sync-video {str(new_settings["vsync"]).lower()}')
                        if hasattr(window, 'entity'):
                            props = WindowProperties()
                            # Toggle window mode to force vsync update
                            current_props = window.entity.getProperties()
                            props.setFullscreen(not current_props.getFullscreen())
                            window.entity.requestProperties(props)
                            invoke(lambda: window.entity.requestProperties(WindowProperties().setFullscreen(new_settings["fullscreen"])), delay=0.1)
                        settings_applied.append("vsync")
                        logger.info(f"Applied V-Sync: {new_settings['vsync']}")
                    
                    # 3. FOV already applied in real-time via slider
                    # (camera.fov is updated as slider moves)
                    
                    # 4. Sound volumes (prepared for future audio system)
                    if new_settings["sound_volume"] != settings["sound_volume"]:
                        settings_applied.append("sound volume")
                        logger.info(f"Sound volume set to: {new_settings['sound_volume']}")
                    
                    if new_settings["music_volume"] != settings["music_volume"]:
                        settings_applied.append("music volume")
                        logger.info(f"Music volume set to: {new_settings['music_volume']}")
                    
                except Exception as e:
                    logger.error(f"Failed to apply settings: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Show message (stay in settings menu)
                if settings_applied:
                    self.show_message(f"Settings saved and applied!\n({', '.join(settings_applied)})")
                else:
                    self.show_message("Settings saved!")
            
            def go_back():
                logger.info("Settings: Back button pressed")
                self.show_main_menu()
            
            # Position buttons side by side - Back on left, Apply on right
            # Back button (left side)
            back_texture = menu_textures["buttons"].get("back")
            back_button = Button(
                parent=self.root,
                model="quad",
                texture=back_texture,
                scale=(0.35, 0.1),  # Smaller size
                position=(-0.25, -0.38),  # Left side
                origin=(0, 0),
                color=color.white,
                highlight_color=color.white,
                pressed_color=color.white,
                text="",
            )
            back_button.on_click = go_back
            self.widgets.append(back_button)
            
            # Apply/Done button (right side)
            done_texture = menu_textures["buttons"].get("done")
            done_button = Button(
                parent=self.root,
                model="quad",
                texture=done_texture,
                scale=(0.35, 0.1),  # Smaller size
                position=(0.25, -0.38),  # Right side
                origin=(0, 0),
                color=color.white,
                highlight_color=color.white,
                pressed_color=color.white,
                text="",
            )
            done_button.on_click = save_and_back
            self.widgets.append(done_button)

        def _discover_worlds(self) -> list[str]:
            if not WORLD_SAVE_DIR.exists():
                return []
            names = []
            for path in sorted(WORLD_SAVE_DIR.glob("*")):
                if path.is_dir():
                    names.append(path.name)
            return names

        def show_message(self, message: str) -> None:
            if self.info_text is not None:
                destroy(self.info_text)
            self.info_text = Text(
                parent=self.root,
                text=f"<bold>{message}</bold>",
                origin=(0, 0),
                position=(0, 0.42),
                scale=1.3,
                color=color.black,
            )
            self.info_text.thickness = 2.5
            self.widgets.append(self.info_text)

        def start_game(self, world_info: Optional[Dict[str, Any]] = None) -> None:
            # Prevent multiple calls
            if hasattr(self, '_game_starting') and self._game_starting:
                logger.warning("start_game called while already starting, ignoring duplicate call")
                return
            
            self._game_starting = True
            logger.info("Starting game with world info")
            
            self.root.disable()
            self.clear_widgets()
            if world_info is None:
                world_info = self.pending_world_info

            if world_info is None:
                logger.warning("No world info available to start the game")
                self._game_starting = False
                return

            self.pending_world_info = world_info
            self.on_start_game(world_info)
            
            # Reset flag after a short delay
            invoke(lambda: setattr(self, '_game_starting', False), delay=1.0)

        def hide(self) -> None:
            self.root.disable()

        def show(self) -> None:
            self.root.enable()

        def show_loading_screen(self, world_info):
            """Display a loading screen while generating the world."""
            # Create loading screen background
            self.pending_world_info = world_info
            self.selected_mode = world_info["metadata"].get("mode", "creative")
            self.loading_root = Entity(parent=camera.ui, z=-10)
            
            # Black background
            black_bg = Entity(
                parent=self.loading_root,
                model="quad",
                scale=(camera.aspect_ratio * 2, 2),
                color=color.rgb(0, 0, 0),
                z=0
            )
            
            # GUI background texture (fullscreen)
            gui_bg = Entity(
                parent=self.loading_root,
                model="quad",
                texture="assets/menu/GUIS/Gui_grey.png",
                scale=(camera.aspect_ratio * 2, 2),
                color=color.white,
                z=0.1
            )
            
            # Title text - now with z=-1 to be in front
            title_text = Text(
                parent=self.loading_root,
                text="Generating World",
                position=(0, 0.1),
                origin=(0, 0),
                scale=2.5,
                color=color.white,
                thickness=2,
                bold=True,
                z=-1
            )
            
            # World name text
            world_name_text = Text(
                parent=self.loading_root,
                text=world_info["metadata"]["display_name"],
                position=(0, 0.05),
                origin=(0, 0),
                scale=1.5,
                color=color.rgb(180, 180, 180),
                thickness=1.5,
                z=-1
            )
            
            # Loading status text with animated dots
            self.loading_status = Text(
                parent=self.loading_root,
                text="Building terrain",
                position=(0, -0.05),
                origin=(0, 0),
                scale=1.2,
                color=color.rgb(200, 200, 200),
                z=-1
            )
            
            # Progress bar background
            progress_bg = Entity(
                parent=self.loading_root,
                model="quad",
                scale=(0.6, 0.03),
                position=(0, -0.15),
                color=color.rgb(40, 40, 40),
                origin=(0, 0),
                z=-1
            )
            
            # Progress bar fill
            self.progress_bar = Entity(
                parent=self.loading_root,
                model="quad",
                scale=(0.01, 0.025),
                position=(-0.3, -0.15),
                color=color.rgb(100, 200, 100),
                origin=(-0.5, 0),
                z=-2
            )
            
            # Animation state
            self.loading_time = 0
            self.loading_dots = 0
            self.loading_stage = 0
            self.loading_start_time = time.time()
            self.loading_complete = False
            self.loading_stages = [
                "Building terrain",
                "Placing blocks",
                "Growing trees",
                "Finalizing world"
            ]
            
            # Update function for loading animation
            def update_loading():
                # Stop if already complete
                if self.loading_complete or not hasattr(self, 'loading_root') or self.loading_root is None:
                    return
                
                elapsed = time.time() - self.loading_start_time
                self.loading_time += time.dt
                
                # Update dots animation (every 0.5 seconds)
                if self.loading_time > 0.5:
                    self.loading_time = 0
                    self.loading_dots = (self.loading_dots + 1) % 4
                    dots = "." * self.loading_dots
                    
                    # Progress through stages based on elapsed time
                    stage_duration = 0.75  # seconds per stage
                    self.loading_stage = min(int(elapsed / stage_duration), len(self.loading_stages) - 1)
                    
                    if hasattr(self, 'loading_status') and self.loading_status:
                        self.loading_status.text = self.loading_stages[self.loading_stage] + dots
                
                # Animate progress bar smoothly
                progress = min(elapsed / 3.0, 1.0)
                if hasattr(self, 'progress_bar') and self.progress_bar:
                    self.progress_bar.scale_x = 0.6 * progress
                
                # Complete loading after 3 seconds
                if elapsed >= 3.0 and not self.loading_complete:
                    self.loading_complete = True
                    # Clean up loading screen elements
                    if hasattr(self, 'loading_root') and self.loading_root:
                        try:
                            # Remove the update function FIRST to prevent it being called again
                            self.loading_root.update = None
                            self.loading_root.enabled = False
                            self.loading_root.visible = False
                            destroy(self.loading_root)
                        except Exception as e:
                            logger.warning(f"Error destroying loading screen: {e}")
                        self.loading_root = None
                    # Clear references to prevent memory leaks
                    self.loading_status = None
                    self.progress_bar = None
                    # Start the game immediately using invoke to defer to next frame
                    invoke(lambda: self.start_game(world_info), delay=0.1)
            
            self.loading_root.update = update_loading
        
        def show_create_world_dialog(self) -> None:
            logger.info("show_create_world_dialog() called")
            if self.dialog_root is not None:
                logger.info("Destroying existing dialog_root")
                destroy(self.dialog_root)

            logger.info("Creating new dialog_root")
            self.dialog_root = Entity(parent=camera.ui)
            self.world_name_active = True
            dialog = Entity(
                parent=self.dialog_root,
                model="quad",
                texture=menu_textures["panels"].get("dialog"),
                scale=(0.8, 0.65),
                position=(0, 0),
            )

            prompt = Text(
                parent=self.dialog_root,
                text="World Name:",
                origin=(-0.5, 0),
                position=(-0.25, 0.15),
                scale=1.2,
                color=color.black,
                thickness=2,
            )

            # Use Ursina's built-in InputField for reliable text input
            self.world_name_input = InputField(
                parent=self.dialog_root,
                default_value="New World",
                max_lines=1,
                character_limit=18,  # Stricter character limit to fit the box
                position=(0, 0.05),
                text_color=color.rgb(255, 255, 255),
                active=True,
                z=-1  # Behind buttons
            )
            
            # Style the input field with visible background
            self.world_name_input.color = color.rgb(60, 60, 60)
            self.world_name_input.highlight_color = color.rgb(80, 80, 80)
            
            # Scale the ENTIRE InputField entity (not internals) to make it larger
            # This keeps the internal coordinate system intact
            self.world_name_input.scale_x = 0.6
            self.world_name_input.scale_y = 0.1
            
            # Track selected gamemode
            self.selected_gamemode = "survival"
            
            # Gamemode buttons with textures
            def create_gamemode_button(mode_name, texture_key, x_pos):
                btn = Button(
                    parent=self.dialog_root,
                    model="quad",
                    texture=menu_textures["buttons"].get(texture_key),
                    scale=(0.25, 0.08),
                    position=(x_pos, -0.1),
                    color=color.white if mode_name == self.selected_gamemode else color.rgb(150, 150, 150),
                    highlight_color=color.rgb(200, 200, 200),
                )
                
                def select_mode():
                    self.selected_gamemode = mode_name
                    # Update button colors
                    for button_info in gamemode_buttons:
                        if button_info["mode"] == mode_name:
                            button_info["button"].color = color.white
                        else:
                            button_info["button"].color = color.rgb(150, 150, 150)
                
                btn.on_click = select_mode
                return btn
            
            gamemode_buttons = [
                {"mode": "survival", "button": create_gamemode_button("survival", "survival", -0.15)},
                {"mode": "sandbox", "button": create_gamemode_button("sandbox", "sandbox", 0.15)},
            ]
                
            # Set up Enter key submission
            def on_submit():
                create_and_start()
                
            self.world_name_input.on_submit = on_submit

            def create_and_start():
                logger.info("create_and_start() called")
                
                # Deactivate input field FIRST to prevent it from capturing clicks
                if self.world_name_input is not None:
                    try:
                        self.world_name_input.active = False
                    except:
                        pass
                
                name = "New World"  # Default name
                if self.world_name_input is not None:
                    try:
                        text_value = str(self.world_name_input.text).strip()
                        logger.info(f"Read world name from input: '{text_value}'")
                        # Ensure name doesn't exceed limit even if field somehow allows it
                        if len(text_value) > 18:
                            text_value = text_value[:18]
                        # Use the entered name if it's not empty
                        if text_value:
                            name = text_value
                        logger.info(f"Creating world with name: '{name}'")
                    except Exception as e:
                        logger.warning(f"Error reading world name input: {e}, using default 'New World'")
                        name = "New World"
                else:
                    logger.warning("World name input is None, using default 'New World'")
                
                # Get selected gamemode
                gamemode = self.selected_gamemode if hasattr(self, 'selected_gamemode') else "survival"
                
                # Close the create world dialog
                destroy(self.dialog_root)
                self.dialog_root = None
                self.world_name_input = None
                
                # Create world with selected gamemode and show loading screen
                world_info = create_new_world(name, gamemode)
                logger.info("Created world directory: %s with gamemode: %s", world_info["path"], gamemode)
                self.selected_mode = world_info["metadata"].get("mode", "creative")
                self.refresh_worlds()

                # Show loading screen before starting game
                self.pending_world_info = world_info
                self.show_loading_screen(world_info)

            def cancel_dialog():
                logger.info("Create World cancelled")
                # Deactivate input field first to prevent conflicts
                if self.world_name_input is not None:
                    try:
                        self.world_name_input.active = False
                    except:
                        pass
                    self.world_name_input = None
                if self.dialog_root is not None:
                    destroy(self.dialog_root)
                    self.dialog_root = None

            def dialog_input(key):
                if key == "escape":
                    # Only handle escape if the input field is not active
                    if self.world_name_input is None or not self.world_name_input.active:
                        cancel_dialog()
                    else:
                        # Deactivate input field on first escape
                        self.world_name_input.active = False

            create_button = Button(
                parent=self.dialog_root,
                model="quad",
                texture=menu_textures["buttons"].get("create_world_small"),
                scale=(0.35, 0.12),
                position=(0.18, -0.24),
                origin=(0, 0),
                color=color.white,
                z=-2  # In front of input field to capture clicks
            )
            create_button.on_click = create_and_start

            cancel_button = Button(
                parent=self.dialog_root,
                model="quad",
                texture=menu_textures["buttons"].get("cancel"),
                scale=(0.35, 0.12),
                position=(-0.18, -0.24),
                origin=(0, 0),
                color=color.white,
                z=-2  # In front of input field to capture clicks
            )
            cancel_button.on_click = cancel_dialog

            self.dialog_root.input = dialog_input

    # Track if world is currently being started
    _world_starting = False
    
    def start_world(world_info: Dict[str, Any]) -> None:
        nonlocal player_entity, current_world_info, current_player_position, current_player_orientation, current_hotbar_slot, enable_face_culling, _world_starting
        
        # Prevent multiple simultaneous calls
        if _world_starting:
            logger.warning("start_world called while already starting, ignoring duplicate call")
            return
        
        _world_starting = True
        logger.info("Starting world creation/loading")
        
        clear_world()
        world_path = world_info.get("path")
        metadata = world_info.get("metadata", {})
        mode = metadata.get("mode", "creative")

        if world_path is None:
            logger.warning("World info missing path; generating default world")
            current_world_info = world_info
            generate_flat_world(width=10, depth=10, block_size=1.0, mark_dirty=False)
            current_player_position = None
            current_player_orientation = None
            current_hotbar_slot = 0
        else:
            world_path = Path(world_path)
            current_world_info = world_info
            saved_blocks, saved_player_pos, saved_orientation, saved_hotbar_slot = load_world_state(world_path)

            if saved_blocks:
                logger.info("Loading %d blocks from saved state", len(saved_blocks))
                # Disable face culling during block loading
                enable_face_culling = False
                current_world_blocks.clear()
                for key, block_type in saved_blocks.items():
                    position = key_to_pos(key)
                    textures = block_type_textures.get(block_type, face_textures)
                    Block(
                        position=position,
                        block_size=1.0,
                        textures=textures,
                        block_type=block_type,
                        record_state=True,
                        mark_dirty=False,
                    )
                # Enable face culling after loading
                enable_face_culling = True
                reset_dirty_flag()
                current_player_position = saved_player_pos
                current_player_orientation = saved_orientation
                current_hotbar_slot = saved_hotbar_slot
            else:
                logger.info("No saved world state found; generating new terrain")
                generate_flat_world(width=10, depth=10, block_size=1.0, mark_dirty=False)
                current_player_position = None
                current_player_orientation = None
                current_hotbar_slot = 0

        logger.info("Generated world: 10x10 blocks")

        # Update spawn height to account for the 3-layer world (1 grass + 2 dirt)
        # Grass is at y=2.5, so spawn player at y=4.0 (1.5 blocks above grass)
        spawn_height = 4.0

        # Create the player based on game mode
        if mode == "survival":
            player = SurvivalController(
                position=Vec3(0, spawn_height, 0),
                x=0,
            )
            logger.info("Starting survival mode controller")
        else:  # Default to creative mode
            player = CreativeController(
                position=Vec3(0, spawn_height, 0),
                x=0,
            )
            logger.info("Starting creative mode controller")

        if current_player_position is not None:
            player.position = current_player_position

        if current_player_orientation is not None:
            yaw, pitch = current_player_orientation
            player.rotation_y = yaw
            camera_pivot = getattr(player, "camera_pivot", None)
            if camera_pivot is not None:
                camera_pivot.rotation_x = pitch
            else:
                player.rotation_x = pitch
        
        # Restore hotbar selection
        if hasattr(player, 'hotbar') and hasattr(player, 'selected_slot'):
            player.selected_slot = current_hotbar_slot
            player.hotbar.select_slot(current_hotbar_slot)
            selected_item = player.hotbar.get_selected_item()
            if selected_item:
                player.selected_block_type = selected_item

        # Create player visuals
        player_body = Entity(
            parent=player,
            model='cube',
            color=color.rgb(100, 150, 200),  # Light blue color
            scale=(0.4, 0.8, 0.3),  # Width, height, depth
            position=(0, -0.5, 0),  # Position below camera
            collider=None,  # No separate collider, parent handles collision
            visible=False,
        )

        # Add a simple head
        player_head = Entity(
            parent=player,
            model='cube',
            color=color.rgb(220, 180, 140),  # Skin tone
            scale=(0.35, 0.35, 0.35),
            position=(0, 0.1, 0),  # Above the camera
            collider=None,
            visible=False,
        )

        player_entity = player
        logger.info(
            "Controls: WASD to move, mouse to look, SPACE/SHIFT to fly up/down, ESC to quit"
        )
        
        # Reset flag after world is fully loaded
        _world_starting = False
        logger.info("World start complete")

    def return_to_main_menu() -> None:
        logger.info("Returning to main menu")
        save_current_world_state(force=True)
        clear_world()
        resume_game(lock_mouse=False)
        menu_manager.show_main_menu()

    def pause_options_placeholder() -> None:
        logger.info("Pause options clicked (not implemented)")

    pause_menu = PauseMenu(
        on_main_menu=return_to_main_menu,
        on_options=pause_options_placeholder,
        on_resume=resume_game,
    )

    menu_manager = MenuManager(on_start_game=start_world)

    logger.info("Starting Ursina app loop")
    try:
        app.run()
    except Exception as exc:
        logger.exception("Fatal error during Ursina run loop: %s", exc)
        raise


if __name__ == "__main__":
    main()
