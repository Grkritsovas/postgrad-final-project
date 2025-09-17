import pandas as pd
import json
import time
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Set
from tqdm.auto import tqdm

try:
    from rapidfuzz.fuzz import ratio as fuzzy_ratio
except ImportError:
    fuzzy_ratio = None
    logging.warning("rapidfuzz not found. Fuzzy matching will be less effective. Run: pip install rapidfuzz")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PUNCT_RE = re.compile(r"[’'`´]")
PARENS_RE = re.compile(r"[\(\[].*?[\)\]]")
FEAT_RE = re.compile(r"\b(feat|ft|featuring)\b.*", re.I)

class LyricsFetcher:
    """
    A class to fetch and clean song lyrics using the Genius API,
    with robust caching, progress tracking, and sophisticated matching logic.
    """
    def __init__(self, genius_api_token: str, cache_path: str = "lyrics_cache.json", fuzzy_threshold: int = 85):
        self.genius = self._init_genius(genius_api_token)
        self.cache_path = Path(cache_path)
        self.cache = self._load_cache()
        self.fuzzy_threshold = fuzzy_threshold
        logger.info(f"LyricsFetcher initialized with fuzzy matching threshold of {self.fuzzy_threshold}.")

    def _init_genius(self, token: str):
        try:
            import lyricsgenius
            client = lyricsgenius.Genius(token, skip_non_songs=True, remove_section_headers=True, timeout=15, retries=3)
            client.verbose = False
            return client
        except ImportError:
            logger.error("lyricsgenius package not found. Run: pip install lyricsgenius")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Genius API: {e}")
            raise

    def _load_cache(self) -> Dict[str, Optional[str]]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode cache file at {self.cache_path}. Starting fresh.")
        return {}

    def _save_cache(self):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache to {self.cache_path}: {e}")

    def _clean_lyrics(self, raw_lyrics: Optional[str]) -> Optional[str]:
        if not raw_lyrics or not isinstance(raw_lyrics, str): return None
        if re.search(r"\b(instrumental)\b", raw_lyrics, re.I): return "instrumental"
        
        text = raw_lyrics
        text = re.sub(r'^.*?Read More.*?(?=\n|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
        patterns_to_remove = [
            r'^\d+\s*Contributors.*?Lyrics', r'\[.*?\]', r'Embed\s*$', r'You might also like.*$',
            r'https?://\S+', r'See.*?Live.*?Get tickets.*?', r'^.*?Lyrics\s*',
        ]
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text.strip() if len(text.strip()) >= 50 else None

    def _normalize_name(self, name: str) -> str:
        if not name: return ""
        s = name.lower()
        s = FEAT_RE.sub("", s)
        s = PUNCT_RE.sub("", s)
        s = re.sub(r"&", " and ", s)
        s = re.sub(r"[^a-z0-9\s]+", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _normalize_title(self, title: str) -> str:
        if not title: return ""
        s = PARENS_RE.sub("", title)
        return self._normalize_name(s)

    # More robust artist and title variant generation.
    def _artist_variants(self, artist: str) -> Set[str]:
        base = self._normalize_name(artist)
        parts = base.split()
        variants = {base}
        try:
            and_idx = parts.index("and")
            variants.add(" ".join(parts[:and_idx]))
        except ValueError:
            pass
        if len(parts) > 1:
            variants.add(" ".join(parts[:-1]))
        return {v for v in variants if v}

    def _title_variants(self, title: str) -> Set[str]:
        base = self._normalize_title(title)
        variants = {base}
        if ":" in base:
            variants.add(base.split(":", 1)[0].strip())
        return {v for v in variants if v}

    def _is_artist_match(self, search_artist: str, found_artist: str) -> bool:
        sa_tokens = set(self._normalize_name(search_artist).split())
        fa_tokens = set(self._normalize_name(found_artist).split())
        if not sa_tokens: return False
        if sa_tokens.issubset(fa_tokens) or fa_tokens.issubset(sa_tokens): return True
        if len(sa_tokens & fa_tokens) >= max(1, len(sa_tokens) - 1): return True
        return fuzzy_ratio is not None and fuzzy_ratio(" ".join(sa_tokens), " ".join(fa_tokens)) >= self.fuzzy_threshold

    def _is_title_match(self, search_title: str, found_title: str) -> bool:
        st = self._normalize_title(search_title)
        ft = self._normalize_title(found_title)
        if not st: return False
        if st in ft or ft in st: return True
        return fuzzy_ratio is not None and fuzzy_ratio(st, ft) >= self.fuzzy_threshold

    # In src/lyrics_utils.py, use this as the _fetch_single_song method

    def _fetch_single_song(self, artist: str, track: str) -> Optional[str]:
        """Robustly fetches lyrics using a multi-pass strategy with detailed error logging."""
        logger.debug(f"Searching for: '{artist}' - '{track}'")
        artist_vars = self._artist_variants(artist)
        title_vars = self._title_variants(track)
        
        # Pass 1: Search with artist constraint
        for a_var in artist_vars:
            for t_var in title_vars:
                try:
                    song = self.genius.search_song(t_var, a_var)
                    if song and self._is_artist_match(artist, song.artist) and self._is_title_match(track, song.title):
                        lyrics = self._clean_lyrics(song.lyrics)
                        if lyrics: return lyrics
                except Exception as e:
                    logger.error(f"API Error during Pass 1 for '{t_var}' by '{a_var}': {e}")
                    time.sleep(15) # Sleep longer on error

        # Pass 2: Title-only search with post-validation
        for t_var in title_vars:
            try:
                song = self.genius.search_song(t_var)
                if song and self._is_artist_match(artist, song.artist) and self._is_title_match(track, song.title):
                    lyrics = self._clean_lyrics(song.lyrics)
                    if lyrics: return lyrics
            except Exception as e:
                logger.error(f"API Error during Pass 2 for '{t_var}': {e}")
                time.sleep(15)

        return None
        
    def enrich_dataframe(self, df: pd.DataFrame, artist_col: str = "artist_name", track_col: str = "track_name", batch_save_size: int = 50) -> pd.DataFrame:
        df_copy = df.copy()
        if 'lyrics' not in df_copy.columns:
            df_copy['lyrics'] = None

        rows_to_process = df_copy[df_copy['lyrics'].isna()]
        logger.info(f"Found {len(rows_to_process)} songs needing lyrics.")

        pbar = tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Fetching Lyrics")
        
        processed_count = 0
        for idx, row in pbar:
            artist = str(row[artist_col])
            track = str(row[track_col])
            cache_key = f"{artist.lower()}|{track.lower()}"

            if cache_key in self.cache:
                df_copy.at[idx, 'lyrics'] = self.cache[cache_key]
            else:
                lyrics = self._fetch_single_song(artist, track)
                self.cache[cache_key] = lyrics
                df_copy.at[idx, 'lyrics'] = lyrics
                processed_count += 1
                time.sleep(3)

                if processed_count > 0 and processed_count % batch_save_size == 0:
                    self._save_cache()
                    logger.info(f"Cache saved after processing {processed_count} new songs.")
        
        self._save_cache()
        logger.info("Lyrics fetching complete. Final cache saved.")
        
        found_count = df_copy['lyrics'].notna().sum()
        logger.info(f"Found lyrics for {found_count} out of {len(df_copy)} songs.")
        return df_copy