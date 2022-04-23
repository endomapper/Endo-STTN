for d in */ ; do base=$(basename "$d") ; cd $base ; zip -r $base * ; mv "${base}.zip" .. ; cd .. ; done;

