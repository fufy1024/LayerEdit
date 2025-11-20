from edit.main import  edit, inverse, init_model

input_image = "./example/1/init_image.png"
mask_paths = ["./example/1/mask_1.png", "./example/1/mask_2.png"]
source_prompt = "a right bird and a left bird"
fg_prompt = ["a crochet bird", "a origami bird"]


NUM_DDIM_STEPS = 50
edit_path  = './output/edit.png'    
save_path = './output/save.png'
pipe, inversion, device = init_model(
    model_path="stabilityai/stable-diffusion-xl-base-1.0",
    num_ddim_steps=NUM_DDIM_STEPS,
)
latent_list, latent, prompt_str, prompt_embeds, pooled_prompt_embeds = inverse(
    inversion, input_image, source_prompt=source_prompt
)
edit(
    pipe=pipe,
    steps=NUM_DDIM_STEPS,
    image_path=input_image,
    mask_paths=mask_paths,
    fg_prompt=fg_prompt,
    bg_prompt=prompt_str,
    bg_negative=prompt_str,
    latent=latent,
    latent_list=latent_list,
    edit_path=edit_path,
    save_path=save_path,
    device=device,
)
