import os
import yt_dlp
from pydub import AudioSegment

# Crear carpeta de destino
os.makedirs("data/raw", exist_ok=True)

# Lista de canciones (Nombre y URL)
songs = [
    #  Rock
    ("queen_bohemian_rhapsody", "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"),
    ("nirvana_smells_like_teen_spirit", "https://www.youtube.com/watch?v=hTWKbfoikeg"),
    ("the_rolling_stones_paint_it_black", "https://www.youtube.com/watch?v=O4irXQhgMqg"),
    ("ac_dc_back_in_black", "https://www.youtube.com/watch?v=pAgnJDJN4VA"),
    ("guns_n_roses_sweet_child_o_mine", "https://www.youtube.com/watch?v=1w7OgIMMRc4"),

    #  Electrónica
    ("daft_punk_one_more_time", "https://www.youtube.com/watch?v=FGBhQbmPwH8"),
    ("deadmau5_strobe", "https://www.youtube.com/watch?v=tuICgIoz6fY"), ## error
    ("the_prodigy_breathe", "https://www.youtube.com/watch?v=XMxuihx834I"), ## error
    ("calvin_harris_summer", "https://www.youtube.com/watch?v=ebXbLfLACGM"),
    ("avicii_levels", "https://www.youtube.com/watch?v=_ovdm2yX4MA"),

    #  Hip-Hop / Rap
    ("eminem_lose_yourself", "https://www.youtube.com/watch?v=_Yhyp-_hX2s"),
    ("kendrick_lamar_humble", "https://www.youtube.com/watch?v=tvTRZJ-4EyI"),
    ("50_cent_in_da_club", "https://www.youtube.com/watch?v=5qm8PH4xAss"),
    ("drake_gods_plan", "https://www.youtube.com/watch?v=xpVfcZ0ZcFM"),
    ("travis_scott_sicko_mode", "https://www.youtube.com/watch?v=6ONRf7h3Mdk"),

    #  Pop
    ("michael_jackson_billie_jean", "https://www.youtube.com/watch?v=Zi_XLOBDo_Y"),
    ("taylor_swift_shake_it_off", "https://www.youtube.com/watch?v=nfWlot6h_JM"),
    ("dua_lipa_dont_start_now", "https://www.youtube.com/watch?v=oygrmJFKYZY"),
    ("ed_sheeran_shape_of_you", "https://www.youtube.com/watch?v=JGwWNGJdvx8"),
    ("adele_rolling_in_the_deep", "https://www.youtube.com/watch?v=rYEDA3JcQqw"),

    #  Regional Mexicano
    ("vicente_fernandez_el_rey", "https://www.youtube.com/watch?v=3lT8R1g8F98"), # error
    ("pedro_infante_cien_anos", "https://www.youtube.com/watch?v=2WjV5hQrUqY"), # error
    ("los_tigres_del_norte_la_puerta_negra", "https://www.youtube.com/watch?v=QjWZ2g_MCmg"), # error
    ("banda_ms_el_color_de_tus_ojos", "https://www.youtube.com/watch?v=BG3zuQ5b2jU"), # error
    ("christian_nodal_adios_amor", "https://www.youtube.com/watch?v=H6G4-Fvi6j8"), # error

    #  Indie / Alternativo
    ("arctic_monkeys_do_i_wanna_know", "https://www.youtube.com/watch?v=bpOSxM0rNPM"),
    ("tame_impala_the_less_i_know_the_better", "https://www.youtube.com/watch?v=go_3mmz3ZCM"), # error
    ("radiohead_creep", "https://www.youtube.com/watch?v=XFkzRNyygfk"),
    ("the_strokes_reptilia", "https://www.youtube.com/watch?v=b8-tXG8KrWs"),
    ("florence_the_machine_dog_days_are_over", "https://www.youtube.com/watch?v=iWOyfLBYtuU"),

    #  Reggaetón / Urbano
    ("bad_bunny_titi_me_pregunto", "https://www.youtube.com/watch?v=Jh4QFaPmdss"),
    ("j_balvin_mi_gente", "https://www.youtube.com/watch?v=wnJ6LuUFpMo"),
    ("daddy_yankee_gasolina", "https://www.youtube.com/watch?v=7zp1TbLFPp8"),
    ("karol_g_tusa", "https://www.youtube.com/watch?v=tbneQDc2H3I"),
    ("don_omar_danza_kuduro", "https://www.youtube.com/watch?v=7zp1TbLFPp8"),

    #  Clásica
    ("beethoven_symphony_no5", "https://www.youtube.com/watch?v=fOk8Tm815lE"),
    ("mozart_eine_kleine_nachtmusik", "https://www.youtube.com/watch?v=oy2zDJPIgwc"),
    ("tchaikovsky_swan_lake", "https://www.youtube.com/watch?v=9cNQFB0TDfY"),
    ("vivaldi_four_seasons_spring", "https://www.youtube.com/watch?v=6LAPFM3dgag"),
    ("debussy_clair_de_lune", "https://www.youtube.com/watch?v=CvFH_6DNRCY"),

    #  Jazz
    ("miles_davis_so_what", "https://www.youtube.com/watch?v=zqNTltOGh5c"),
    ("john_coltrane_my_favorite_things", "https://www.youtube.com/watch?v=qWG2dsXV5HI"),
    ("dave_brubeck_take_five", "https://www.youtube.com/watch?v=tT9Eh8wNMkw"),
    ("louis_armstrong_what_a_wonderful_world", "https://www.youtube.com/watch?v=CWzrABouyeE"), # error
    ("nina_simone_feeling_good", "https://www.youtube.com/watch?v=D5Y11hwjMNs"),

    #  Otros / Misceláneos
    ("bts_dynamite", "https://www.youtube.com/watch?v=gdZLi9oWNZg"),
    ("shakira_hips_dont_lie", "https://www.youtube.com/watch?v=DUT5rEU6pqM"),
    ("abba_dancing_queen", "https://www.youtube.com/watch?v=xFrGuyw1V8s"),
    ("bob_marley_no_woman_no_cry", "https://www.youtube.com/watch?v=IT8XvzIfi5Y"), # error
    ("rammstein_du_hast", "https://www.youtube.com/watch?v=W3q8Od5qJio"),
]

# Opciones para yt-dlp
def download_audio(title, url):
    output_path = f"data/raw/{title}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Descargando: {title}")
            ydl.download([url])
        except Exception as e:
            print(f"Error con {title}: {e}")

# Descargar canciones
for name, link in songs:
    download_audio(name, link)
    
