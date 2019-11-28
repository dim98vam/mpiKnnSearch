# ####################################################################
#
#			   C/C++ Makefile
#
# 
#
# Adapted from
#  http://www.cs.swarthmore.edu/~newhall/unixhelp/howto_makefiles.html
#
# ####################################################################
#
# 
# 'make lib'		build the libraries .a
# 'make clean'  removes all .o and executable files
#

# define the shell to bash
SHELL := /bin/bash

# define the C/C++ compiler to use,default here is clang
CC = mpicc

# setup paths
SRCDIR = src
INCDIR = inc
LIBDIR = lib


# define compile-time flags
CFLAGS = -Wall 

# define any directories containing header files
INCLUDES = -I $(INCDIR)

# define the source file for the library
SRC = knnring

# define the different possible executables
TYPES = mpi 


#
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above
#

# call everytime
.PRECIOUS: $(LIBDIR)/%.a


lib: $(addprefix $(LIBDIR)/, $(addsuffix .a, $(addprefix $(SRC)_, $(TYPES))))


# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .cpp file) and $@: the name of the target of the rule (a .o file)

$(LIBDIR)/$(SRC)_%.a: $(SRCDIR)/$(SRC)_%.o
	mkdir -p $(LIBDIR)
	ar rcs $@ $<

# (see the gnu make manual section about automatic variables)
$(SRCDIR)/$(SRC)_%.o: $(SRCDIR)/$(SRC)_%.c
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ -c $< -lopenblas -lm

clean:
	$(RM) $(LIBDIR)/* $(SRCDIR)/*~ $(SRCDIR)/*.o $(INCDIR)/*~ *~