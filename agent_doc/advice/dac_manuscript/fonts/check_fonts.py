#!/usr/bin/env python3
"""Verify exact bundled bytes, actual faces and Fontconfig matches; no substitutes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--check',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'MANIFEST.json').read_text())
    for rel,item in manifest['files'].items():
        path=ROOT/rel
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:
            raise SystemExit('Missing/modified font dependency (aliases forbidden): '+str(path))
    if args.prepare:
        cache=args.prepare.resolve();cache.mkdir(parents=True,exist_ok=True)
        (cache/'fonts.conf').write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig>'+''.join('<dir>'+escape(str(ROOT/name))+'</dir>' for name in ('urw','noto','lm-math','liberation'))+'<cachedir>'+escape(str(cache/'fontconfig'))+'</cachedir><include ignore_missing="yes">/etc/fonts/fonts.conf</include><alias binding="strong"><family>Arial</family><prefer><family>Liberation Sans</family></prefer></alias></fontconfig>\n')
        (cache/'fontconfig').mkdir(exist_ok=True)
    if args.check:
        if not os.environ.get('FONTCONFIG_FILE'):raise SystemExit('Source fonts/env.sh first')
        for rel,item in manifest['files'].items():
            if 'family' not in item:continue
            rows=subprocess.check_output(['fc-scan','--format','%{index}\t%{family}\t%{style}\t%{postscriptname}\n',str(ROOT/rel)],text=True).splitlines()
            row=next((x.split('\t') for x in rows if x.split('\t')[0]==str(item['index'])),None)
            if row is None or row[1].split(',')[0]!=item['family'] or row[2].split(',')[0]!=item['style'] or row[3]!=item['postscript']:
                raise SystemExit('Incorrect internal font identity: '+rel)
            matched=subprocess.check_output(['fc-match','--format','%{file}\t%{family}\t%{style}\t%{postscriptname}',item['family']+':style='+item['style']],text=True).split('\t')
            if (len(matched)!=4 or not Path(matched[0]).resolve().is_relative_to(ROOT)
                    or item['family'] not in matched[1].split(',') or matched[3]!=item['postscript']):
                raise SystemExit('Substituted requested face '+item['family']+' '+item['style']+': '+repr(matched))
        print('Verified exact bundled bytes and requested regular/bold/italic faces; no undeclared family substitution (Arial explicitly maps to Liberation Sans).')

if __name__=='__main__':main()
