#! /bin/sh

# items=("covid icu" "hospital" "oxygen" "concentrator" "ambulance" "ventilator")
items=("covid icu")

# TODO Use no retweet, language translate, etc in twint

for i in "${items[@]}"; do
    echo "$i"
    ~/.local/bin/twint --near Bangalore --since 2021-05-01 --search "$i" --json -o input.json >/dev/null
done
