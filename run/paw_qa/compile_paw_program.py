import programasweights as paw

program_just_compiled = paw.compile(
    "Answer questions about the current run: " \
    "its trigger mode, who conducted it, its date, "
    "and similar run properties."
)

paw.function(program_just_compiled.id)

print("New program compiled with ID:", program_just_compiled.id)
