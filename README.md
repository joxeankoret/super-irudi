# Super Irudi

Super Irudi is a command line based tool which aims to help automating some common tasks for enhancing photographs.

# Example usages

For example, let's say that you have one photograph with nice colors and another one with not that good colors and you want to 'copy' the color histogram from one to another, one can run a command line like the following one:
```
$ ./super_irudi.py m74-bad-core.jpg -i -mh m51-good-core.jpg
[Sat May 14 16:10:46 2022] Matching histograms...
[Sat May 14 16:10:47 2022] Source image's size (4014, 3865)
[Sat May 14 16:10:49 2022] Reference's image size (11553, 7633)
[Sat May 14 16:11:01 2022] Displaying final image
```

It will show the following preview: ![M74 as M51](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-as-m51.png)

As we can see, the resulted image is much better, as the core of the M74 galaxy is now clearer. We could remove the command line flag `-i` (used to display previews in an interactive way) and add, at the end, the command line option `-o m74-fixed.jpg` to save this image. But before doing so we're going to apply one little filter: the "winter" filter, that, basically, increases the blue color in the image. In order to do so, we would issue a command line like the following:

```
$ ./super_irudi.py m74-bad-core.jpg -mh m51-good-core.jpg -i --winter
```
![M74 as M51 with the winter filter](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-winter.png)

Notice that I have moved the 'interactive' flag right before the `--winter` command line flag because, otherwise, it would show 2 previews: one for each modification we apply to the picture. There is another method, instead, to see only the final version of the image instead of each step, as with the `-i` flag: adding at the end the `-s` (show) flag:

```
$ ./super_irudi.py m74-bad-core.jpg -mh m51-good-core.jpg --winter -s
```

![At the left, the original M74; at the right, the final version](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-winter-diff-original.png).

As we can see, however, the flag `-s` shows the differences with the original picture, not against the previous step (in order to do so, use the `-i` flag to enable/disable interactiveness for this or that step). By the way, naturally, there is a `--summer` filter that we can use like we have done with the `--winter` one:

```
$ ./super_irudi.py m74-bad-core.jpg -mh m51-good-core.jpg --summer -s
```

![M74 as M51 with a warmer filter](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-summer.png).

Now, let's apply to it a Gaussian blur filter just for the sake of doing so. The new command line will be the following:

```
./super_irudi.py m74-bad-core.jpg -mh m51-good-core.jpg --summer -gb=4 -s
```

![M74 as M51, warmer and blurred](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-summer-gb.png).

Please note that I have used the value '4' for the Gaussian blur, but it's possible to use other values (1, 5, 10, whatever). Now, let's mirror the picture:

```
./super_irudi.py m74-bad-core.jpg -mh m51-good-core.jpg --summer -gb=4 -m -s
```

![M74 as M51, warmer, blurred and mirrored](https://github.com/joxeankoret/super-irudi/blob/main/examples/m74-summer-gb.png).

And, now, supposing this is the final state that we want this picture to be, we can remove the `-s` flag and append `-o m74-final.jpg` to save this picture. But, more importantly, now let's say that we have a hundred such pictures: you can use this tool and this command line to produce a hundred exactly equal images in the command line, in your automation for whatever process you have, etc...

